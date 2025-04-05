import os

def process_file(file_path):
    max_value = float('-inf')
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line and get the first number
                numbers = line.strip().split()
                if numbers:  # Check if line is not empty
                    try:
                        first_number = float(numbers[0])
                        max_value = max(max_value, first_number)
                    except ValueError:
                        continue  # Skip if first element is not a number
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    return max_value

def update_file(file_path, value):
    try:
        with open(file_path, 'w') as file:
            file.write(str(value))
    except Exception as e:
        print(f"Error updating {file_path}: {str(e)}")

def main():
    directory = "data_set"
    
    # Process all .txt files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            file_max = process_file(file_path)
            if file_max != float('-inf'):  # Only update if we found a valid number
                update_file(file_path, file_max)
                print(f"Updated {filename} with value: {file_max}")
            else:
                print(f"No valid numbers found in {filename}")

if __name__ == "__main__":
    main() 