import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json 


# --- HÀM TÍNH TRỌNG SỐ ---
def get_class_weights(data_dir):
    categories = sorted(os.listdir(data_dir))
    # Đếm số lượng ảnh trong mỗi class
    counts = [len(os.listdir(os.path.join(data_dir, c))) for c in categories]
    total = sum(counts)
    num_classes = len(counts)
    
    # Công thức: weight = total / (num_classes * count)
    weights = {i: total / (num_classes * count) for i, count in enumerate(counts)}
    return weights, categories

def stratified_split(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Chia dữ liệu theo tỷ lệ và đảm bảo tỷ lệ các lớp (Stratified)
    """
    # Tạo cấu trúc thư mục đích
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Lấy danh sách tất cả file và nhãn (tên thư mục con)
    all_files = []
    labels = []
    
    source_path = Path(source_dir)
    for class_folder in os.listdir(source_path):
        class_path = source_path / class_folder
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                all_files.append(class_path / file_name)
                labels.append(class_folder)
                # Tạo thư mục lớp trong các tập split
                os.makedirs(os.path.join(output_dir, 'train', class_folder), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'val', class_folder), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'test', class_folder), exist_ok=True)

    # Bước 1: Split ra Train và (Val + Test)
    # Tỷ lệ Val+Test = 1 - Train
    val_test_ratio = val_ratio + test_ratio
    train_files, val_test_files, train_labels, val_test_labels = train_test_split(
        all_files, labels, test_size=val_test_ratio, stratify=labels, random_state=42
    )

    # Bước 2: Split (Val + Test) thành Val và Test
    # Tính lại tỷ lệ để chia tiếp
    relative_test_ratio = test_ratio / val_test_ratio
    val_files, test_files, _, _ = train_test_split(
        val_test_files, val_test_labels, test_size=relative_test_ratio, stratify=val_test_labels, random_state=42
    )

    # Hàm copy file
    def copy_files(files, split_name):
        for file_path in files:
            class_name = file_path.parent.name
            dest_path = os.path.join(output_dir, split_name, class_name, file_path.name)
            shutil.copy2(file_path, dest_path)

    print("--- ĐANG PHÂN CHIA DỮ LIỆU... ---")
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    print(f"--- XONG! DỮ LIỆU ĐÃ ĐƯỢC CHIA TẠI: {output_dir} ---")


    # --- TÍNH VÀ LƯU TRỌNG SỐ TẠI ĐÂY ---
    print("--- ĐANG TÍNH TOÁN TRỌNG SỐ LỚP (CLASS WEIGHTS)... ---")
    train_dir = os.path.join(output_dir, 'train')
    weights, categories = get_class_weights(train_dir)
    
    with open('class_weights.json', 'w') as f:
        json.dump(weights, f)
    print(f"--- ĐÃ LƯU TRỌNG SỐ VÀO 'class_weights.json' ---")

if __name__ == "__main__":
    stratified_split('data', 'data_final')