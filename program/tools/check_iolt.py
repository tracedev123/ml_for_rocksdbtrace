def is_line_defective(line):
    count = 0
    for char in line:
        if char == ':':
            count += 1
    return count < 5 or count > 7
def main():
    file_name = "iolt.txt"
    try:
        with open(file_name, 'r') as file:
            line_number = 0
            for line in file:
                line_number += 1
                if is_line_defective(line):
                    print(f"Defective line at line number {line_number}: {line}", end='')
    except FileNotFoundError:
        print("Error opening file")
if __name__ == "__main__":
    main()