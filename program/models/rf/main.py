from natsort import natsorted
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
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
feature_name = 'HISTORY' ########################################################
test_ratio_in_percentage = 20 ########################################################
test_ratio = test_ratio_in_percentage / 100 ########################################################
table = PrettyTable()
table.field_names = ["Location", "Average Score"]
for csv_file in all_csv_files_sorted:
    file_path = os.path.join(root_folder, csv_file)
    data = pd.read_csv(file_path)
    data = data.head(int(np.round(len(data) * 0.001)))
    data = data.reset_index()
    test_size = int(np.round(len(data) * test_ratio))
    train_data, test_data = data[:-test_size], data[-test_size:]
    num_columns = train_data.shape[1]
    prediction_indices = range(len(train_data), len(train_data) + test_size)
    prediction_data = pd.DataFrame(prediction_indices, columns=['index'], index=prediction_indices)
    for col in [column for column in train_data.columns if train_data[column].nunique() == 1]:
        prediction_data[col] = pd.Series(train_data[col].iloc[0]).repeat(test_size).values
    columns_not_in_prediction = [col for col in train_data.columns.tolist() if col not in prediction_data.columns.tolist()]
    unique_values_dict = {}
    for col in columns_not_in_prediction:
        unique_values = train_data[col].nunique()
        unique_values_dict[col] = unique_values
    continuous_columns = []
    discrete_columns = []
    for col, unique_values in unique_values_dict.items():
        if unique_values > 500:
            continuous_columns.append(col)
        else:
            discrete_columns.append(col)
    num_past_measurements = 20
    start_index = len(data) - test_size
    for col in columns_not_in_prediction:
        for i in range(1, num_past_measurements + 1):
            data[f'{col}_{i}'] = data[col].shift(i * 25)
            if data[f'{col}_{i}'].dtype == 'int64':
                data[f'{col}_{i}'].fillna(-1, inplace=True)
            elif data[f'{col}_{i}'].dtype == 'float64':
                data[f'{col}_{i}'].fillna(-1.0, inplace=True)
            if data[f'{col}_{i}'].dtype != data[col].dtype:
                data[f'{col}_{i}'] = data[f'{col}_{i}'].astype(data[col].dtype)
            train_data.loc[:start_index, f'{col}_{i}'] = data.loc[:start_index, f'{col}_{i}']
            test_data.loc[:start_index, f'{col}_{i}'] = data.loc[:start_index, f'{col}_{i}']
            prediction_data.loc[start_index:, f'{col}_{i}'] = data.loc[start_index:, f'{col}_{i}']
    num_rows = int(np.round(len(train_data) * 0.05))
    for col in columns_not_in_prediction:
        features = [f'{col}_{i}' for i in range(1, num_columns)]
        X_test = prediction_data[features]
        if col in continuous_columns:
            X_train = train_data[features].values[-num_rows:]
            y_train = train_data[col].values[-num_rows:]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            X_train = train_data[features]
            y_train = train_data[col]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred < -1, -1, y_pred)
        if train_data[col].dtype == np.int64:
            y_pred = np.round(y_pred).astype(np.int64)
        prediction_data[col] = y_pred
    columns_to_drop = ['index']
    if columns_not_in_prediction:
        start_index = train_data.columns.get_loc(f'{columns_not_in_prediction[0]}_1')
        end_index = train_data.columns.get_loc(f'{columns_not_in_prediction[-1]}_{num_past_measurements}')
        columns_to_drop.extend(data.columns[start_index:end_index + 1])
    data = data.drop(columns_to_drop, axis=1)
    train_data = train_data.drop(columns_to_drop, axis=1)
    test_data = test_data.drop(columns_to_drop, axis=1)
    prediction_data = prediction_data.drop(columns_to_drop, axis=1)
    prediction_data = prediction_data[train_data.columns]
    scores = {}
    for col in prediction_data.columns:
        if col in continuous_columns:
            score = r2_score(test_data[col], prediction_data[col])
            if score < 0:
                score = 0
            metric = 'R-squared'
        else:
            score = accuracy_score(test_data[col], prediction_data[col])
            metric = 'Accuracy'
        scores[col] = (score, metric)
    average_score = np.mean([score for score, metric in scores.values()])
    average_score = round(average_score * 100, 2)
    table.add_row([file_path, f'{average_score}%'])
    table.add_row([''] * len(table.field_names))
output_file_path = f'{feature_name}_{test_ratio_in_percentage}.txt'
with open(output_file_path, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    print(table)
    sys.stdout = original_stdout
print(f'Table saved to: {output_file_path}')
############################################################################################################################
############################################################################################################################
############################################################################################################################
feature_name = 'HISTORY' ########################################################
test_ratio_in_percentage = 50 ########################################################
test_ratio = test_ratio_in_percentage / 100 ########################################################
table = PrettyTable()
table.field_names = ["Location", "Average Score"]
for csv_file in all_csv_files_sorted:
    file_path = os.path.join(root_folder, csv_file)
    data = pd.read_csv(file_path)
    data = data.head(int(np.round(len(data) * 0.001)))
    data = data.reset_index()
    test_size = int(np.round(len(data) * test_ratio))
    train_data, test_data = data[:-test_size], data[-test_size:]
    num_columns = train_data.shape[1]
    prediction_indices = range(len(train_data), len(train_data) + test_size)
    prediction_data = pd.DataFrame(prediction_indices, columns=['index'], index=prediction_indices)
    for col in [column for column in train_data.columns if train_data[column].nunique() == 1]:
        prediction_data[col] = pd.Series(train_data[col].iloc[0]).repeat(test_size).values
    columns_not_in_prediction = [col for col in train_data.columns.tolist() if col not in prediction_data.columns.tolist()]
    unique_values_dict = {}
    for col in columns_not_in_prediction:
        unique_values = train_data[col].nunique()
        unique_values_dict[col] = unique_values
    continuous_columns = []
    discrete_columns = []
    for col, unique_values in unique_values_dict.items():
        if unique_values > 500:
            continuous_columns.append(col)
        else:
            discrete_columns.append(col)
    num_past_measurements = 20
    start_index = len(data) - test_size
    for col in columns_not_in_prediction:
        for i in range(1, num_past_measurements + 1):
            data[f'{col}_{i}'] = data[col].shift(i * 25)
            if data[f'{col}_{i}'].dtype == 'int64':
                data[f'{col}_{i}'].fillna(-1, inplace=True)
            elif data[f'{col}_{i}'].dtype == 'float64':
                data[f'{col}_{i}'].fillna(-1.0, inplace=True)
            if data[f'{col}_{i}'].dtype != data[col].dtype:
                data[f'{col}_{i}'] = data[f'{col}_{i}'].astype(data[col].dtype)
            train_data.loc[:start_index, f'{col}_{i}'] = data.loc[:start_index, f'{col}_{i}']
            test_data.loc[:start_index, f'{col}_{i}'] = data.loc[:start_index, f'{col}_{i}']
            prediction_data.loc[start_index:, f'{col}_{i}'] = data.loc[start_index:, f'{col}_{i}']
    num_rows = int(np.round(len(train_data) * 0.05))
    for col in columns_not_in_prediction:
        features = [f'{col}_{i}' for i in range(1, num_columns)]
        X_test = prediction_data[features]
        if col in continuous_columns:
            X_train = train_data[features].values[-num_rows:]
            y_train = train_data[col].values[-num_rows:]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            X_train = train_data[features]
            y_train = train_data[col]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred < -1, -1, y_pred)
        if train_data[col].dtype == np.int64:
            y_pred = np.round(y_pred).astype(np.int64)
        prediction_data[col] = y_pred
    columns_to_drop = ['index']
    if columns_not_in_prediction:
        start_index = train_data.columns.get_loc(f'{columns_not_in_prediction[0]}_1')
        end_index = train_data.columns.get_loc(f'{columns_not_in_prediction[-1]}_{num_past_measurements}')
        columns_to_drop.extend(data.columns[start_index:end_index + 1])
    data = data.drop(columns_to_drop, axis=1)
    train_data = train_data.drop(columns_to_drop, axis=1)
    test_data = test_data.drop(columns_to_drop, axis=1)
    prediction_data = prediction_data.drop(columns_to_drop, axis=1)
    prediction_data = prediction_data[train_data.columns]
    scores = {}
    for col in prediction_data.columns:
        if col in continuous_columns:
            score = r2_score(test_data[col], prediction_data[col])
            if score < 0:
                score = 0
            metric = 'R-squared'
        else:
            score = accuracy_score(test_data[col], prediction_data[col])
            metric = 'Accuracy'
        scores[col] = (score, metric)
    average_score = np.mean([score for score, metric in scores.values()])
    average_score = round(average_score * 100, 2)
    table.add_row([file_path, f'{average_score}%'])
    table.add_row([''] * len(table.field_names))
output_file_path = f'{feature_name}_{test_ratio_in_percentage}.txt'
with open(output_file_path, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    print(table)
    sys.stdout = original_stdout
print(f'Table saved to: {output_file_path}')
############################################################################################################################
############################################################################################################################
############################################################################################################################
feature_name = 'HISTORY' ########################################################
test_ratio_in_percentage = 80 ########################################################
test_ratio = test_ratio_in_percentage / 100 ########################################################
table = PrettyTable()
table.field_names = ["Location", "Average Score"]
for csv_file in all_csv_files_sorted:
    file_path = os.path.join(root_folder, csv_file)
    data = pd.read_csv(file_path)
    data = data.head(int(np.round(len(data) * 0.001)))
    data = data.reset_index()
    test_size = int(np.round(len(data) * test_ratio))
    train_data, test_data = data[:-test_size], data[-test_size:]
    num_columns = train_data.shape[1]
    prediction_indices = range(len(train_data), len(train_data) + test_size)
    prediction_data = pd.DataFrame(prediction_indices, columns=['index'], index=prediction_indices)
    for col in [column for column in train_data.columns if train_data[column].nunique() == 1]:
        prediction_data[col] = pd.Series(train_data[col].iloc[0]).repeat(test_size).values
    columns_not_in_prediction = [col for col in train_data.columns.tolist() if col not in prediction_data.columns.tolist()]
    unique_values_dict = {}
    for col in columns_not_in_prediction:
        unique_values = train_data[col].nunique()
        unique_values_dict[col] = unique_values
    continuous_columns = []
    discrete_columns = []
    for col, unique_values in unique_values_dict.items():
        if unique_values > 500:
            continuous_columns.append(col)
        else:
            discrete_columns.append(col)
    num_past_measurements = 20
    start_index = len(data) - test_size
    for col in columns_not_in_prediction:
        for i in range(1, num_past_measurements + 1):
            data[f'{col}_{i}'] = data[col].shift(i * 25)
            if data[f'{col}_{i}'].dtype == 'int64':
                data[f'{col}_{i}'].fillna(-1, inplace=True)
            elif data[f'{col}_{i}'].dtype == 'float64':
                data[f'{col}_{i}'].fillna(-1.0, inplace=True)
            if data[f'{col}_{i}'].dtype != data[col].dtype:
                data[f'{col}_{i}'] = data[f'{col}_{i}'].astype(data[col].dtype)
            train_data.loc[:start_index, f'{col}_{i}'] = data.loc[:start_index, f'{col}_{i}']
            test_data.loc[:start_index, f'{col}_{i}'] = data.loc[:start_index, f'{col}_{i}']
            prediction_data.loc[start_index:, f'{col}_{i}'] = data.loc[start_index:, f'{col}_{i}']
    num_rows = int(np.round(len(train_data) * 0.05))
    for col in columns_not_in_prediction:
        features = [f'{col}_{i}' for i in range(1, num_columns)]
        X_test = prediction_data[features]
        if col in continuous_columns:
            X_train = train_data[features].values[-num_rows:]
            y_train = train_data[col].values[-num_rows:]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            X_train = train_data[features]
            y_train = train_data[col]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred < -1, -1, y_pred)
        if train_data[col].dtype == np.int64:
            y_pred = np.round(y_pred).astype(np.int64)
        prediction_data[col] = y_pred
    columns_to_drop = ['index']
    if columns_not_in_prediction:
        start_index = train_data.columns.get_loc(f'{columns_not_in_prediction[0]}_1')
        end_index = train_data.columns.get_loc(f'{columns_not_in_prediction[-1]}_{num_past_measurements}')
        columns_to_drop.extend(data.columns[start_index:end_index + 1])
    data = data.drop(columns_to_drop, axis=1)
    train_data = train_data.drop(columns_to_drop, axis=1)
    test_data = test_data.drop(columns_to_drop, axis=1)
    prediction_data = prediction_data.drop(columns_to_drop, axis=1)
    prediction_data = prediction_data[train_data.columns]
    scores = {}
    for col in prediction_data.columns:
        if col in continuous_columns:
            score = r2_score(test_data[col], prediction_data[col])
            if score < 0:
                score = 0
            metric = 'R-squared'
        else:
            score = accuracy_score(test_data[col], prediction_data[col])
            metric = 'Accuracy'
        scores[col] = (score, metric)
    average_score = np.mean([score for score, metric in scores.values()])
    average_score = round(average_score * 100, 2)
    table.add_row([file_path, f'{average_score}%'])
    table.add_row([''] * len(table.field_names))
output_file_path = f'{feature_name}_{test_ratio_in_percentage}.txt'
with open(output_file_path, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    print(table)
    sys.stdout = original_stdout
print(f'Table saved to: {output_file_path}')