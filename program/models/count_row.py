from natsort import natsorted
from prettytable import PrettyTable
import numpy as np
import os
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')
root_folder = 'C:/dse/datasets'
def find_and_sort_csv_files(folder):
    csv_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    csv_files_sorted = natsorted(csv_files)
    return csv_files_sorted
all_csv_files_sorted = find_and_sort_csv_files(root_folder)
table = PrettyTable()
table.field_names = ["Location", "Row Count"]
for csv_file in all_csv_files_sorted:
    file_path = os.path.join(root_folder, csv_file)
    data = pd.read_csv(file_path)
    data = data.head(int(np.round(len(data) * 0.001)))
    table.add_row([file_path, len(data)])
    table.add_row([''] * len(table.field_names))
code_path = __file__
output_file_name = os.path.splitext(os.path.basename(code_path))[0]
output_file_path = f'{output_file_name}.txt'
with open(output_file_path, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    print(table)
    sys.stdout = original_stdout
print(f'Table saved to: {output_file_path}')