import pandas as pd

def reverse_cssd(file_path):
    """
    Reads a CSV file, reverses the 'CSSD' column, and saves the modified data back to the same file.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Check if the 'CSSD' column exists
        if 'CSSD' not in df.columns:
            print(f"Error: 'CSSD' column not found in {file_path}")
            return

        # Reverse the 'CSSD' column
        #df['CSSD'] = df['CSSD'] * -1
        """max_num = df['CSSD'].max() + 1
        df['CSSD'] = max_num - df['CSSD']"""
        df['WR'] = 1 - df['WR']

        # Save the modified DataFrame back to the CSV file, overwriting the original
        df.to_csv("reverse_30_data.csv", index=False)  # index=False prevents saving the index column

        print(f"Successfully reversed 'CSSD' in {file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
file_path = "new_30_data.csv" # Replace with your csv file's path
reverse_cssd(file_path)