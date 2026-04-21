import tensorflow as tf
import os
from src.pipeline import get_pipeline
from src.model_builder_v2 import build_cnn_model

def run_smoke_test():
    print("--- BẮT ĐẦU CHẠY THỬ (SMOKE TEST) ---")
    
    # 1. Kiểm tra thư mục dữ liệu
    data_dir = 'data_final'
    if not os.path.exists(data_dir):
        print(f"LỖI: Không tìm thấy thư mục {data_dir}. Hãy đảm bảo bạn đã chạy script stratified_split.")
        return

    # 2. Nạp dữ liệu (Pipeline)
    print("-> Đang nạp dữ liệu từ data_final...")
    try:
        train_ds = get_pipeline(os.path.join(data_dir, 'train'), is_training=True)
        val_ds = get_pipeline(os.path.join(data_dir, 'val'), is_training=False)
        print("-> Nạp dữ liệu thành công!")
    except Exception as e:
        print(f"LỖI khi nạp dữ liệu: {e}")
        return

    # 3. Khởi tạo mô hình
    print("-> Đang khởi tạo mô hình...")
    try:
        model = build_cnn_model(input_shape=(224, 224, 3), num_classes=4) 
        model.summary()
        print("-> Khởi tạo mô hình thành công!")
    except Exception as e:
        print(f"LỖI khi tạo mô hình: {e}")
        return

    # 4. Chạy thử 2 epochs
    print("\n-> Bắt đầu huấn luyện thử (2 epochs)...")
    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=2  # Chỉ chạy 2 vòng để test nhanh
        )
        print("\n--- TEST THÀNH CÔNG! HỆ THỐNG CỦA BẠN ĐÃ SẴN SÀNG. ---")
    except Exception as e:
        print(f"LỖI trong quá trình huấn luyện: {e}")

if __name__ == "__main__":
    run_smoke_test()