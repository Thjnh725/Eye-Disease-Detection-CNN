import os
import shutil
import concurrent.futures
from PIL import Image
from sklearn.model_selection import train_test_split

def check_data_balance(data_dir):
    """Đếm số lượng tệp trong từng lớp, tính toán phần trăm và cảnh báo mất cân bằng."""
    print("\n PHÂN TÍCH DỮ LIỆU (EDA) & CÂN BẰNG LỚP")
    print("-" * 50)
    counts = {}
    total = 0
    categories = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for cat in categories:
        cat_path = os.path.join(data_dir, cat)
        num_files = len(os.listdir(cat_path))
        counts[cat] = num_files
        total += num_files
        
    for cat, count in counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"Lớp {cat:<10}: {count:>5} ảnh ({percentage:.2f}%)")
        
    avg = total / len(categories) if categories else 1
    max_count = max(counts.values()) if counts else 0
    if max_count > avg * 1.5:
        print("\n⚠️ CẢNH BÁO: Dữ liệu đang bị MẤT CÂN BẰNG nghiêm trọng!")
    else:
        print("\n✅ Dữ liệu tương đối cân bằng.")
    return categories

def get_class_weights(data_dir):
    """Tính toán trọng số cho từng lớp để đối xử công bằng giữa các lớp bệnh."""
    categories = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    counts = [len(os.listdir(os.path.join(data_dir, c))) for c in categories]
    total = sum(counts)
    num_classes = len(counts)
    
    # Công thức: weight = total / (num_classes * count)
    weights = {i: total / (num_classes * count) if count > 0 else 1.0 for i, count in enumerate(counts)}
    print(f"\n⚖️ Trọng số tự động (Class Weights): {weights}")
    return weights, categories

def stratified_split(source_dir, output_dir):
    """Phân chia dữ liệu Stratified (80/10/10) đảm bảo phân bổ đồng đều."""
    print(f"\n✂️ TIẾN HÀNH CHIA DỮ LIỆU (STRATIFIED SPLIT 80/10/10) TỪ: {source_dir}")
    
    all_files = []
    labels = []
    categories = sorted(os.listdir(source_dir))
    
    for class_idx, category in enumerate(categories):
        cat_path = os.path.join(source_dir, category)
        if not os.path.isdir(cat_path): continue
            
        os.makedirs(os.path.join(output_dir, "train", category), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val", category), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", category), exist_ok=True)
        
        for file in os.listdir(cat_path):
            all_files.append(os.path.join(cat_path, file))
            labels.append(category)

    # Bước 1: Tách 80% Train, 20% (Val + Test)
    train_files, val_test_files, train_labels, val_test_labels = train_test_split(
        all_files, labels, test_size=0.20, stratify=labels, random_state=42
    )
    
    # Bước 2: Tách 20% đó thành 10% Val và 10% Test
    val_files, test_files, _, _ = train_test_split(
        val_test_files, val_test_labels, test_size=0.50, stratify=val_test_labels, random_state=42
    )
    
    # Hàm xử lý làm sạch bằng Pillow và copy SIÊU TỐC (Dùng Đa luồng - Multi-threading)
    def process_and_copy(files_list, split_name):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        
        def process_single_file(src_path):
            if not src_path.lower().endswith(valid_exts):
                return
                
            class_name = os.path.basename(os.path.dirname(src_path))
            file_name = os.path.basename(src_path)
            dest_path = os.path.join(output_dir, split_name, class_name, file_name)
            
            try:
                # 1. Verify bằng Pillow (không tốn RAM)
                with Image.open(src_path) as img:
                    img.verify()
                
                # 2. Đọc lại và Ép về RGB, ghi đè
                with Image.open(src_path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(dest_path)
            except Exception as e:
                print(f"⚠️ Loại bỏ tệp hỏng/lỗi: {file_name} ({e})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(process_single_file, files_list))

    process_and_copy(train_files, "train")
    process_and_copy(val_files, "val")
    process_and_copy(test_files, "test")
    print(f"✅ Đã chia và làm sạch dữ liệu thành công vào: {output_dir}")