"""
MarketingIQ data pipeline.
Reads the raw campaign CSV and produces data.js for the dashboard frontend.

Expected folder layout (relative to this script):
    marketingiq/
      data/    <- tech_advertising_campaigns_dataset.csv goes here
      scripts/ <- this file lives here
      site/    <- index.html lives here; data.js is written here

Run from inside the scripts/ folder:
    python build_data.py
"""
import pandas as pd
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'data', 'tech_advertising_campaigns_dataset.csv')
OUT = os.path.join(HERE, '..', 'site', 'data.js')

df = pd.read_csv(SRC)
df['start_date'] = pd.to_datetime(df['start_date'])
df['month'] = df['start_date'].dt.strftime('%Y-%m')

campaign_cols = {
    'campaign_id': 'id', 'platform': 'platform', 'campaign_objective': 'objective',
    'device_type': 'device', 'target_audience_age': 'age', 'target_audience_gender': 'gender',
    'industry_vertical': 'vertical', 'budget_tier': 'budget', 'retargeting_flag': 'retargeting',
    'creative_format': 'creative', 'creative_emotion': 'emotion', 'ad_placement': 'placement',
    'income_bracket': 'income', 'month': 'month',
    'ad_spend': 'spend', 'revenue': 'revenue', 'profit': 'profit',
    'ROAS': 'roas', 'CPA': 'cpa', 'conversion_rate': 'convrate', 'CTR': 'ctr',
    'clicks': 'clicks', 'impressions': 'impressions', 'conversions': 'conversions',
}
camp_df = df[list(campaign_cols.keys())].rename(columns=campaign_cols)
for col in ['spend', 'revenue', 'profit', 'roas', 'cpa', 'convrate', 'ctr']:
    camp_df[col] = camp_df[col].round(2)
campaigns = camp_df.to_dict(orient='records')

def agg_by(col):
    g = df.groupby(col).agg(
        spend=('ad_spend', 'sum'), revenue=('revenue', 'sum'), profit=('profit', 'sum'),
        conversions=('conversions', 'sum'), clicks=('clicks', 'sum'), impressions=('impressions', 'sum')
    ).reset_index()
    g['roas'] = (g['revenue'] / g['spend']).round(3)
    g['cpa'] = (g['spend'] / g['conversions']).round(2)
    g['convrate'] = (g['conversions'] / g['clicks'] * 100).round(2)
    g['ctr'] = (g['clicks'] / g['impressions'] * 100).round(3)
    for c in ['spend', 'revenue', 'profit']:
        g[c] = g[c].round(2)
    return g.rename(columns={col: 'name'}).to_dict(orient='records')

aggregates = {
    'platform': agg_by('platform'), 'placement': agg_by('ad_placement'), 'age': agg_by('target_audience_age'),
    'device': agg_by('device_type'), 'gender': agg_by('target_audience_gender'), 'income': agg_by('income_bracket'),
    'creative': agg_by('creative_format'), 'emotion': agg_by('creative_emotion'), 'vertical': agg_by('industry_vertical'),
    'budget_tier': agg_by('budget_tier'), 'objective': agg_by('campaign_objective'),
}

monthly = df.groupby('month').agg(revenue=('revenue', 'sum'), spend=('ad_spend', 'sum')).reset_index()
monthly['revenue'] = monthly['revenue'].round(2)
monthly['spend'] = monthly['spend'].round(2)
monthly = monthly.to_dict(orient='records')

df['age_bucket'] = pd.cut(df['creative_age_days'], bins=[0, 15, 30, 45, 60, 90], labels=['0-15', '16-30', '31-45', '46-60', '61-90'])
fat = df.groupby('age_bucket', observed=True).agg(clicks=('clicks', 'sum'), impressions=('impressions', 'sum')).reset_index()
fat['ctr'] = (fat['clicks'] / fat['impressions'] * 100).round(2)
fatigue = fat.rename(columns={'age_bucket': 'bucket'})[['bucket', 'ctr']].to_dict(orient='records')

cta = df.groupby('has_call_to_action').agg(clicks=('clicks', 'sum'), impressions=('impressions', 'sum'), conversions=('conversions', 'sum')).reset_index()
cta['ctr'] = (cta['clicks'] / cta['impressions'] * 100).round(2)
cta['convrate'] = (cta['conversions'] / cta['clicks'] * 100).round(2)
cta_impact = cta.rename(columns={'has_call_to_action': 'has_cta'})[['has_cta', 'ctr', 'convrate']].to_dict(orient='records')

totals = {
    'revenue': round(df['revenue'].sum(), 2), 'spend': round(df['ad_spend'].sum(), 2),
    'profit': round(df['profit'].sum(), 2), 'conversions': int(df['conversions'].sum()),
    'clicks': int(df['clicks'].sum()), 'impressions': int(df['impressions'].sum()),
    'roas': round(df['revenue'].sum() / df['ad_spend'].sum(), 3),
    'cpa': round(df['ad_spend'].sum() / df['conversions'].sum(), 2),
    'convrate': round(df['conversions'].sum() / df['clicks'].sum() * 100, 3),
    'ctr': round(df['clicks'].sum() / df['impressions'].sum() * 100, 3),
    'roi_pct': round((df['revenue'].sum() - df['ad_spend'].sum()) / df['ad_spend'].sum() * 100, 1),
    'row_count': len(df),
    'date_min': df['start_date'].min().strftime('%Y-%m-%d'),
    'date_max': df['start_date'].max().strftime('%Y-%m-%d'),
}

output = {
    'totals': totals, 'monthly': monthly, 'aggregates': aggregates,
    'fatigue': fatigue, 'cta_impact': cta_impact, 'campaigns': campaigns,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    f.write('const MIQ_DATA = ')
    json.dump(output, f, separators=(',', ':'))
    f.write(';')

print('Rows:', len(campaigns))
print('File size:', round(os.path.getsize(OUT) / 1024 / 1024, 2), 'MB')
print('Written to:', os.path.abspath(OUT))
