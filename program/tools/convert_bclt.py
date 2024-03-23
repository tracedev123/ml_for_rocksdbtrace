import pandas as pd
df = pd.read_csv('bclt.csv')
df = df.drop('cf_name', axis=1)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df.to_csv('bclt.csv', index=False)