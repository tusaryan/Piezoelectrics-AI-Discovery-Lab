import csv
import os

INPUT_FILE = 'resources/main-datasets/dataset.csv'
OUTPUT_FILE = 'resources/main-datasets/knn_d33.csv'
COLUMN_TO_REMOVE = 'Tc (°C)'

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find '{INPUT_FILE}'. Please run this script from the project root.")
        return

    print(f"Processing '{INPUT_FILE}'...")
    
    with open(INPUT_FILE, 'r', newline='', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:
        
        # We use the built-in csv module to read/write strings directly.
        # This completely avoids any floating-point truncation or precision
        # loss that might occur if using libraries like pandas.
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        try:
            headers = next(reader)
        except StopIteration:
            print("Error: The CSV file is empty.")
            return

        if COLUMN_TO_REMOVE not in headers:
            print(f"Notice: Column '{COLUMN_TO_REMOVE}' not found in the headers.")
            print(f"Available headers: {headers}")
            # If not found, clean up temp file and exit
            os.remove(OUTPUT_FILE)
            return

        # Find the index of the column to remove
        col_index = headers.index(COLUMN_TO_REMOVE)
        
        # Write new headers
        new_headers = headers[:col_index] + headers[col_index+1:]
        writer.writerow(new_headers)
        
        # Process and write all rows
        rows_processed = 0
        for row in reader:
            if len(row) > col_index:
                new_row = row[:col_index] + row[col_index+1:]
                writer.writerow(new_row)
            else:
                # Row is shorter than expected and doesn't reach the column index
                writer.writerow(row)
            rows_processed += 1
            
            if rows_processed % 100000 == 0:
                print(f"Processed {rows_processed} rows...")

    print("\nSuccess!")
    print(f"Total rows processed: {rows_processed}")
    print(f"Removed '{COLUMN_TO_REMOVE}' column.")
    print(f"The updated dataset is saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
