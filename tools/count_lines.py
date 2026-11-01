import os

def count_lines_in_py_files(directory):
    total_lines = 0
    print(f"{'File Name':<30} | {'Line Count':<10}")
    print("-" * 45)
    
    # Lấy danh sách file trong thư mục 'tools'
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    count = len(lines)
                    total_lines += count
                    print(f"{filename:<30} | {count:<10}")
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                
    print("-" * 45)
    print(f"{'Total':<30} | {total_lines:<10}")

if __name__ == "__main__":
    # Chỉ định thư mục là 'tools'
    count_lines_in_py_files('tools')
