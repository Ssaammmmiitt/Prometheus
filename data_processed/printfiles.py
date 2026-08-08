from pathlib import Path

def print_files_no_ext():
    # Set the current directory
    base_dir = Path('.')
    
    # 1. Print files in the CURRENT folder
    print(f"--- Files in Current Folder ---")
    for item in base_dir.iterdir():
        if item.is_file():
            # .stem gives the filename without the extension (e.g., 'image.tif' -> 'image')
            print(item.stem)

    # 2. Print files in specific YEAR subfolders (2018-2025)
    # range(2018, 2026) covers 2018 up to (but not including) 2026
    target_years = range(2018, 2026)

    for year in target_years:
        year_folder = base_dir / str(year)

        # Check if this year folder actually exists
        if year_folder.exists() and year_folder.is_dir():
            print(f"\n--- Files in folder: {year} ---")
            
            for item in year_folder.iterdir():
                if item.is_file():
                    # If you ONLY want to print .tif files, uncomment the line below:
                    # if item.suffix.lower() == '.tif':
                    
                    print(item.stem)

if __name__ == "__main__":
    print_files_no_ext()