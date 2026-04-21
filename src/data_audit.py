import os
from PIL import Image

def audit_dataset(root_dir):
    print(f"\n{'='*20} BẮT ĐẦU QUY TRÌNH KIỂM TRA {'='*20}")
    print(f"Thư mục làm việc: {root_dir}")
    
    total_processed = 0
    total_cleaned = 0
    total_corrupted = 0

    # Lấy danh sách các lớp (thư mục con)
    categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for category in categories:
        cat_path = os.path.join(root_dir, category)
        files = os.listdir(cat_path)
        print(f"\n[CATEGORY] Đang xử lý lớp: {category} (Tổng: {len(files)} ảnh)")
        
        for i, file in enumerate(files):
            filepath = os.path.join(cat_path, file)
            total_processed += 1
            
            # Hiển thị tiến độ xử lý
            print(f"  -> [{i+1}/{len(files)}] {file}...", end="\r") 
            
            try:
                # 1. Kiểm tra tính toàn vẹn
                with Image.open(filepath) as img:
                    img.verify()
                
                # 2. Xử lý chuẩn hóa màu
                with Image.open(filepath) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(filepath, format='JPEG', quality=95)
                        print(f"  -> [THÀNH CÔNG] Chuẩn hóa RGB: {file}")
                        total_cleaned += 1
            
            except Exception as e:
                # 3. Xử lý file hỏng
                print(f"\n  -> [LỖI] File hỏng - Đang xóa: {file} | Lỗi: {e}")
                os.remove(filepath)
                total_corrupted += 1

    print(f"\n{'='*20} TỔNG KẾT QUY TRÌNH {'='*20}")
    print(f"Tổng số ảnh đã kiểm tra: {total_processed}")
    print(f"Số ảnh đã chuyển về RGB: {total_cleaned}")
    print(f"Số ảnh hỏng đã loại bỏ: {total_corrupted}")
    print("=======================================\n")

if __name__ == "__main__":
    audit_dataset('data')