import pandas as pd
def process_file_with_float64_conversion(input_file, output_file):
    header = "time,name,type,operation,latency,io,size,length,offset"
    int64_min = -9223372036854775808
    int64_max = 9223372036854775807
    with open(input_file, 'r') as file:
        lines = file.readlines()
    operation_mapping = {
        'Read': '0', 'Append': '1', 'GetFileSize': '2', 'Close': '3',
        'NewWritableFile': '4', 'PositionedAppend': '5', 'Truncate': '6', 'Prefetch': '7'
    }
    io_mapping = {'OK': '0'}
    type_mapping = {'log': '0', 'sst': '1', 'MANIFEST': '2'}
    processed_lines = [header]
    for line in lines:
        parts = [part.strip() for part in line.split(',')]
        access_time = parts[0].split(':')[-1].strip()
        file_name = parts[1].split(':')[-1].strip()
        file_operation = parts[2].split(':')[-1].strip()
        latency = parts[3].split(':')[-1].strip()
        io_status = parts[4].split(':')[-1].strip()
        size, length, offset = '-1', '-1', '-1'
        for info in parts[5:]:
            key, value = info.split(':')
            key = key.strip()
            value = value.strip()
            if key == 'File Size':
                size = value
            elif key == 'Length':
                length = value
            elif key == 'Offset':
                offset = value
        file_type = 'log' if '.log' in file_name else 'sst' if '.sst' in file_name else 'MANIFEST'
        file_number = file_name.split('.')[0] if '.' in file_name else file_name.split('-')[-1]
        file_number = ''.join(filter(str.isdigit, file_number))
        file_number = str(int(file_number))
        file_operation = operation_mapping.get(file_operation, file_operation)
        io_status = io_mapping.get(io_status, io_status)
        file_type = type_mapping.get(file_type, file_type)
        access_time = float(access_time) if not (int64_min <= int(access_time) <= int64_max) else access_time
        new_line = f"{access_time},{file_number},{file_type},{file_operation},{latency},{io_status},{size},{length},{offset}"
        processed_lines.append(new_line)
    with open(output_file, 'w') as file:
        for line in processed_lines:
            file.write(line + '\n')
input_file_path = 'iolt.txt'
output_file_path = 'iolt.csv'
process_file_with_float64_conversion(input_file_path, output_file_path)
df = pd.read_csv(output_file_path)
df.to_csv(output_file_path, index=False)