import os

def print_directory_tree(start_path):
    print(f"--- Scanning Directory: {start_path} ---\n")
    
    # os.walk() generates the file names in a directory tree
    # by walking the tree either top-down or bottom-up.
    for root, dirs, files in os.walk(start_path):
        
        # Calculate the depth (how many folders deep we are)
        # to determine the indentation
        level = root.replace(start_path, '').count(os.sep)
        indent = '    ' * level
        
        # Print the current folder name
        # os.path.basename returns the last part of the path (the folder name)
        print(f"{indent}|-- 📁 {os.path.basename(root)}/")
        
        # Print all files in this folder
        sub_indent = '    ' * (level + 1)
        for f in files:
            print(f"{sub_indent}|-- {f}")

if __name__ == "__main__":
    # Use '.' for current directory, or replace with a specific path
    target_path = '.' 
    print_directory_tree(target_path)