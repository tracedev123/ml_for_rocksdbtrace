import pandas as pd
name = 'bclt'
data = pd.read_csv(f'{name}.csv')
data_length = len(data)
how_many = 4
chunk_size = data_length // how_many
for i in range(how_many):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < how_many - 1 else data_length
    chunk = data.iloc[start_idx:end_idx]
    chunk.to_csv(f'{name}_{i+1}.csv', index=False)