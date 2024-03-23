import pandas as pd
from io import StringIO
df = pd.read_csv('qlt.txt', sep=" ", header=None, names=["key", "type_id", "cf_id", "value_size", "time"])
df = df[["time", "key", "type_id", "cf_id", "value_size"]]
character = 6
start = 11 + 7 - character
df['key'] = df['key'].apply(lambda x: x[start:start + character])
corrected_csv_file_path = 'qlt.csv'
df['key'] = df['key'].apply(lambda x: int(x, 16))
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df.to_csv(corrected_csv_file_path, index=False)