import pandas as pd
import os

def get_valid_file_input(prompt):
    """
    Get and validate file input from user.
    
    Parameters:
    prompt (str): Message to show user when asking for input
    
    Returns:
    str: Valid file path
    """
    while True:
        file_path = input(prompt)
        if file_path.lower().endswith('.csv'):
            if not os.path.exists(file_path) and 'output' not in prompt.lower():
                print(f"Warning: File '{file_path}' doesn't exist.")
                if input("Would you like to try a different file? (y/n): ").lower() != 'y':
                    break
            else:
                break
        else:
            print("Please enter a CSV file name (must end with .csv)")
    return file_path

def merge_csv_rows():
    """
    Merge rows from three CSV files without any checks or transformations.
    File paths are obtained through user input.
    """
    print("\n=== CSV Row Merger ===")
    print("Please enter the names/paths of your CSV files.")
    
    # Get input files
    base_file = get_valid_file_input("\nEnter the name of the first (base) CSV file: ")
    file2 = get_valid_file_input("Enter the name of the second CSV file: ")
    file3 = get_valid_file_input("Enter the name for the third CSV file: ")
    output_file = get_valid_file_input("Enter the name for the output CSV file: ")
    
    try:
        # Read all CSV files
        print("\nReading files...")
        df1 = pd.read_csv(base_file)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)
        
        # Concatenate dataframes vertically
        print("\nMerging rows from all files...")
        merged_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
        
        # Remove any duplicate rows if desired
        if input("\nWould you like to remove duplicate rows? (y/n): ").lower() == 'y':
            original_rows = len(merged_df)
            merged_df = merged_df.drop_duplicates()
            removed_rows = original_rows - len(merged_df)
            print(f"Removed {removed_rows} duplicate rows.")
        
        # Save to new CSV file
        print("\nSaving merged file...")
        merged_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\nSuccess! Merged file saved as: {output_file}")
        print(f"Total rows in merged file: {len(merged_df)}")
        print(f"Number of columns: {len(merged_df.columns)}")
        print("\nRows from each input file:")
        print(f"Base file: {len(df1)} rows")
        print(f"File 2: {len(df2)} rows")
        print(f"File 3: {len(df3)} rows")
        
        return merged_df
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return None

# Run the program
if __name__ == "__main__":
    merged_data = merge_csv_rows()
