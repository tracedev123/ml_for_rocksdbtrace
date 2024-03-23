import pandas as pd
import re
new_file_path = 'qlt.txt'
df_new = pd.read_csv(new_file_path, sep=" ", header=None, names=["key", "type_id", "cf_id", "value_size", "time"])
df_new = df_new[["time", "key", "type_id", "cf_id", "value_size"]].reset_index(drop=True)
new_corrected_csv_file_path = 'qlt.csv'
df_new.to_csv(new_corrected_csv_file_path, index=False)
df = pd.read_csv(new_corrected_csv_file_path)
def hex_to_string(hex_string):
    modified_hex_string = hex_string[2:]
    byte_data = bytes.fromhex(modified_hex_string)
    result_string = byte_data.decode('latin-1')
    match = re.search(r'user(\d+)', result_string)
    if match:
        user_number = match.group(1)
    else:
        user_number = -1
    return user_number
df['key'] = df['key'].apply(hex_to_string)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df.to_csv(new_corrected_csv_file_path, index=False)