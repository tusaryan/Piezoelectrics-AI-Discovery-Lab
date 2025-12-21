import pandas as pd

def normalize_excel_to_csv(input_file, output_file):
    print(f"Reading file: {input_file}...")
    
    # FIX: Use read_excel for .xlsx files
    # engine='openpyxl' is usually standard for .xlsx
    df = pd.read_excel(input_file, engine='openpyxl')

    # Define translation table for subscripts and superscripts
    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')',
        'ₐ': 'a', 'ₑ': 'e', 'ₒ': 'o', 'ₓ': 'x', 'ₔ': 'h',
        'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₚ': 'p', 'ₛ': 's', 'ₜ': 't'
    }

    superscript_map = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
        'ⁿ': 'n', 'ⁱ': 'i'
    }

    # Combine both maps into a single translation table
    trans_table = str.maketrans({**subscript_map, **superscript_map})

    # Function to apply translation
    def normalize_text(text):
        if isinstance(text, str):
            return text.translate(trans_table)
        return text

    # Apply normalization to all columns
    # Note: If you are on pandas 2.1+, you can use df.map(normalize_text) instead
    df_normalized = df.applymap(normalize_text)

    # Save to CSV
    df_normalized.to_csv(output_file, index=False)
    print(f"Success! Converted to: {output_file}")
    
    return df_normalized

# Usage
if __name__ == "__main__":
    input_filename = "Book2.xlsx" 
    output_filename = "normalized_material_data.csv"
    
    try:
        df_cleaned = normalize_excel_to_csv(input_filename, output_filename)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_filename}'. Please check the file path.")