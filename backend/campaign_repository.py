"""
MarketingIQ data-access layer.

    data_tools.py (business logic: metric formulas, filter meaning, rankings)
        -> CampaignRepository (generic row filtering / aggregation primitives)
            -> DuckDB (primary)  |  Pandas (fallback + correctness reference)

Everything database-specific lives here. Callers pass column names and simple
(column, operator, value) conditions; nothing above this module writes SQL.

Safety: column identifiers are checked against the loaded table's actual schema,
operators against a fixed allowlist, and every value is a bound parameter (`?`),
so no caller-supplied text is ever interpolated into SQL.
"""
import logging
import threading
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger("marketingiq.data")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

TABLE = "campaigns"
SUM_COLUMNS = ("ad_spend", "revenue", "profit", "conversions", "clicks", "impressions")
OPERATORS = {"=": "=", "==": "=", ">": ">", "<": "<", ">=": ">=", "<=": "<="}
DUCKDB_MEMORY_LIMIT = "256MB"


class DataAccessError(Exception):
    """A query failed. The message is safe to surface; details stay in the server log."""

    def __init__(self, message="The analytics query failed."):
        super().__init__(message)


class CampaignRepository(ABC):
    """Conditions are lists of (column, operator, value); operator is one of OPERATORS.
    Aggregate rows use the keys: group (when grouped), campaign_count, and each SUM_COLUMNS name."""
    name = "repository"

    @abstractmethod
    def has_column(self, column: str) -> bool: ...

    @abstractmethod
    def aggregate(self, where: list, group_by: str = None) -> list: ...

    @abstractmethod
    def monthly_sum(self, column: str, where: list) -> list:
        """[(month 'YYYY-MM', sum)] in month order."""

    @abstractmethod
    def numeric_stats(self, columns: list, where: list) -> dict:
        """{column: {min, max, mean, median, p25, p75}} or {column: None} when nothing matches."""

    @abstractmethod
    def find_campaigns(self, where: list, order_by: str, ascending: bool, limit: int, columns: list) -> tuple:
        """(matched_count, [row dicts with `columns`]) — rows sorted, at most `limit`."""

    @abstractmethod
    def bucket_totals(self, where: list, column: str, edges: list, sum_columns: list) -> list:
        """Rows bucketed into right-closed intervals (edges[i], edges[i+1]]; values outside are dropped.
        Returns [(bucket_index, {sum_column: total})] for non-empty buckets, in bucket order."""

    @abstractmethod
    def aggregate_avg(self, columns: list, where: list, group_by: str = None) -> list:
        """Mean of each column over the matching campaigns: [{"group"?, column: mean}] (one row, or one
        per group in group order). For per-campaign averages such as bounce rate."""

    @abstractmethod
    def profile(self, columns: list, date_column: str) -> dict:
        """{"distinct": {column: [non-null values]}, "date_min", "date_max", "row_count"}."""


# ---------------------------------------------------------------------------
# DuckDB (primary)
# ---------------------------------------------------------------------------

class DuckDBCampaignRepository(CampaignRepository):
    """In-memory DuckDB database holding one `campaigns` table, built once at startup.

    One base connection; each worker thread gets its own cursor (DuckDB connections
    are not safe to share across threads, cursors are cheap and share the database)."""
    name = "duckdb"

    def __init__(self, df: pd.DataFrame):
        import duckdb  # imported here so a missing package only disables this backend
        self._duckdb = duckdb
        # Explicit cap: DuckDB's default is 80% of the RAM it detects, which in a small container
        # can be the host's RAM. The whole table is a few MB, so this is generous.
        self._con = duckdb.connect(database=":memory:", config={"memory_limit": DUCKDB_MEMORY_LIMIT})
        self._con.register("_source_df", df)
        self._con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM _source_df")
        self._con.unregister("_source_df")
        self._columns = {r[0] for r in self._con.execute(f"DESCRIBE {TABLE}").fetchall()}
        self._lock = threading.Lock()
        self._local = threading.local()

    # -- plumbing ------------------------------------------------------------
    def _cursor(self):
        cur = getattr(self._local, "cursor", None)
        if cur is None:
            with self._lock:
                cur = self._con.cursor()
            self._local.cursor = cur
        return cur

    def _run(self, sql: str, params: list):
        try:
            return self._cursor().execute(sql, params).fetchall()
        except self._duckdb.Error as e:
            logger.error("duckdb query failed: %s: %s", type(e).__name__, e)
            raise DataAccessError() from e

    def _ident(self, column: str) -> str:
        if column not in self._columns:
            raise DataAccessError(f"Unknown field '{column}'.")
        return '"' + column.replace('"', '""') + '"'

    def _where(self, conditions: list):
        clauses, params = [], []
        for column, op, value in conditions or []:
            if op not in OPERATORS:
                raise DataAccessError(f"Unsupported operator '{op}'.")
            clauses.append(f"{self._ident(column)} {OPERATORS[op]} ?")
            params.append(value)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def has_column(self, column):
        return column in self._columns

    # -- primitives ----------------------------------------------------------
    def aggregate(self, where, group_by=None):
        where_sql, params = self._where(where)
        sums = ", ".join(f"SUM({self._ident(c)})" for c in SUM_COLUMNS)
        if group_by:
            g = self._ident(group_by)
            rows = self._run(f"SELECT {g}, COUNT(*), {sums} FROM {TABLE}{where_sql} GROUP BY 1 ORDER BY 1", params)
            return [{"group": r[0], "campaign_count": r[1], **dict(zip(SUM_COLUMNS, r[2:]))} for r in rows]
        r = self._run(f"SELECT COUNT(*), {sums} FROM {TABLE}{where_sql}", params)[0]
        return [{"campaign_count": r[0], **dict(zip(SUM_COLUMNS, r[1:]))}]

    def monthly_sum(self, column, where):
        where_sql, params = self._where(where)
        return [(m, v) for m, v in self._run(
            f'SELECT "month", SUM({self._ident(column)}) FROM {TABLE}{where_sql} GROUP BY 1 ORDER BY 1', params)]

    def numeric_stats(self, columns, where):
        where_sql, params = self._where(where)
        exprs = []
        for c in columns:
            q = self._ident(c)
            # One list-form quantile call per column (one sort) instead of three.
            exprs.append(f"MIN({q}), MAX({q}), AVG({q}), QUANTILE_CONT({q}, [0.5, 0.25, 0.75]), COUNT({q})")
        row = self._run(f"SELECT {', '.join(exprs)} FROM {TABLE}{where_sql}", params)[0]
        out = {}
        for i, c in enumerate(columns):
            mn, mx, mean, quantiles, n = row[i * 5:(i + 1) * 5]
            out[c] = None if not n else {"min": mn, "max": mx, "mean": mean, "median": quantiles[0],
                                         "p25": quantiles[1], "p75": quantiles[2]}
        return out

    def find_campaigns(self, where, order_by, ascending, limit, columns):
        where_sql, params = self._where(where)
        cols = ", ".join(self._ident(c) for c in columns)
        direction = "ASC" if ascending else "DESC"
        # campaign_id as a tiebreaker makes equal sort values come back in a stable order.
        tiebreak = f", {self._ident('campaign_id')}" if self.has_column("campaign_id") else ""
        rows = self._run(f"SELECT COUNT(*) OVER (), {cols} FROM {TABLE}{where_sql} "
                         f"ORDER BY {self._ident(order_by)} {direction}{tiebreak} LIMIT ?",
                         params + [max(0, int(limit))])
        if not rows:
            return self._run(f"SELECT COUNT(*) FROM {TABLE}{where_sql}", params)[0][0], []
        return rows[0][0], [dict(zip(columns, r[1:])) for r in rows]

    def bucket_totals(self, where, column, edges, sum_columns):
        where_sql, where_params = self._where(where)
        q = self._ident(column)
        cases, case_params = [], []
        for i in range(len(edges) - 1):
            cases.append(f"WHEN {q} > ? AND {q} <= ? THEN {i}")
            case_params += [edges[i], edges[i + 1]]
        sums = ", ".join(f"SUM({self._ident(c)})" for c in sum_columns)
        inner_cols = ", ".join(self._ident(c) for c in sum_columns)
        sql = (f"SELECT bucket, {sums} FROM (SELECT CASE {' '.join(cases)} END AS bucket, {inner_cols} "
               f"FROM {TABLE}{where_sql}) WHERE bucket IS NOT NULL GROUP BY 1 ORDER BY 1")
        return [(r[0], dict(zip(sum_columns, r[1:]))) for r in self._run(sql, case_params + where_params)]

    def aggregate_avg(self, columns, where, group_by=None):
        where_sql, params = self._where(where)
        avgs = ", ".join(f"AVG({self._ident(c)})" for c in columns)
        if group_by:
            g = self._ident(group_by)
            rows = self._run(f"SELECT {g}, {avgs} FROM {TABLE}{where_sql} GROUP BY 1 ORDER BY 1", params)
            return [{"group": r[0], **dict(zip(columns, r[1:]))} for r in rows]
        return [dict(zip(columns, self._run(f"SELECT {avgs} FROM {TABLE}{where_sql}", params)[0]))]

    def profile(self, columns, date_column):
        # LIST keeps NULLs; they are dropped in Python (a FILTER clause here was ~3x slower).
        lists = ", ".join(f"LIST(DISTINCT {self._ident(c)})" for c in columns)
        d = self._ident(date_column)
        row = self._run(f"SELECT {lists}, MIN({d}), MAX({d}), COUNT(*) FROM {TABLE}", [])[0]
        n = len(columns)
        return {"distinct": dict(zip(columns, ([x for x in v if x is not None] for v in row[:n]))),
                "date_min": row[n], "date_max": row[n + 1], "row_count": row[n + 2]}


# ---------------------------------------------------------------------------
# Pandas (fallback, and the reference implementation for correctness tests)
# ---------------------------------------------------------------------------

_PD_OPS = {"=": lambda s, v: s == v, "==": lambda s, v: s == v, ">": lambda s, v: s > v,
           "<": lambda s, v: s < v, ">=": lambda s, v: s >= v, "<=": lambda s, v: s <= v}


class PandasCampaignRepository(CampaignRepository):
    """The original in-memory Pandas approach, behind the same interface."""
    name = "pandas"

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _filtered(self, conditions):
        df = self._df
        for column, op, value in conditions or []:
            if op not in _PD_OPS:
                raise DataAccessError(f"Unsupported operator '{op}'.")
            if column not in df.columns:
                raise DataAccessError(f"Unknown field '{column}'.")
            df = df[_PD_OPS[op](df[column], value)]
        return df

    @staticmethod
    def _sums(df):
        return {"campaign_count": len(df), **{c: df[c].sum() for c in SUM_COLUMNS}}

    def has_column(self, column):
        return column in self._df.columns

    def aggregate(self, where, group_by=None):
        df = self._filtered(where)
        if group_by:
            g = df.groupby(group_by)
            sums, counts = g[list(SUM_COLUMNS)].sum(), g.size()
            return [{"group": name, "campaign_count": int(counts[name]), **row.to_dict()}
                    for name, row in sums.iterrows()]
        return [self._sums(df)]

    def monthly_sum(self, column, where):
        g = self._filtered(where).groupby("month")[column].sum()
        return list(g.items())

    def numeric_stats(self, columns, where):
        df = self._filtered(where)
        out = {}
        for c in columns:
            s = df[c].dropna()
            out[c] = None if s.empty else {"min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median(),
                                           "p25": s.quantile(0.25), "p75": s.quantile(0.75)}
        return out

    def find_campaigns(self, where, order_by, ascending, limit, columns):
        df = self._filtered(where).sort_values(order_by, ascending=ascending)
        return len(df), df[columns].head(max(0, int(limit))).to_dict(orient="records")

    def bucket_totals(self, where, column, edges, sum_columns):
        df = self._filtered(where)
        buckets = pd.cut(df[column], bins=edges, labels=False)
        g = df.groupby(buckets)[list(sum_columns)].sum()
        return [(int(i), row.to_dict()) for i, row in g.iterrows()]

    def aggregate_avg(self, columns, where, group_by=None):
        df = self._filtered(where)
        if group_by:
            g = df.groupby(group_by)[list(columns)].mean()
            return [{"group": name, **row.to_dict()} for name, row in g.iterrows()]
        return [{c: (df[c].mean() if len(df) else None) for c in columns}]

    def profile(self, columns, date_column):
        df = self._df
        return {"distinct": {c: df[c].dropna().unique().tolist() for c in columns},
                "date_min": df[date_column].min(), "date_max": df[date_column].max(), "row_count": len(df)}


def create_repository(df: pd.DataFrame, backend: str = "duckdb") -> CampaignRepository:
    """DuckDB unless told otherwise; falls back to Pandas (logged) if DuckDB can't start."""
    if backend == "pandas":
        return PandasCampaignRepository(df)
    try:
        return DuckDBCampaignRepository(df)
    except Exception as e:
        logger.error("DuckDB unavailable (%s: %s); falling back to the Pandas data backend",
                     type(e).__name__, e)
        return PandasCampaignRepository(df)
