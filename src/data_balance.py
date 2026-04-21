import os

def check_data_balance(data_dir):
    print(f"\n--- KIỂM TRA TÍNH CÂN BẰNG DỮ LIỆU ---")
    categories = sorted(os.listdir(data_dir))
    counts = {c: len(os.listdir(os.path.join(data_dir, c))) for c in categories}
    total = sum(counts.values())
    
    print(f"{'Lớp (Class)':<25} | {'Số lượng':<10} | {'Tỷ lệ (%)':<10}")
    print("-" * 50)
    
    for category, count in counts.items():
        percentage = (count / total) * 100
        print(f"{category:<25} | {count:<10} | {percentage:<10.2f}")
    
    print("-" * 50)
    print(f"Tổng số ảnh: {total}")
    
    # Logic cảnh báo
    max_val = max(counts.values())
    min_val = min(counts.values())
    if max_val / min_val > 1.5:
        print("\nCẢNH BÁO: Dữ liệu đang bị mất cân bằng đáng kể (tỷ lệ lớp lớn nhất/lớp nhỏ nhất > 1.5).")
        print("=> Gợi ý: Hãy sử dụng Class Weights trong quá trình training.")
    else:
        print("\n=> Dữ liệu của bạn khá cân bằng. Bạn có thể yên tâm tiến hành chia dữ liệu.")

if __name__ == "__main__":
    check_data_balance('data')