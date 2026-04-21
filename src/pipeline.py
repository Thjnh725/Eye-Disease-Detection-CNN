import tensorflow as tf
from tensorflow.keras import layers

def get_pipeline(data_dir, batch_size=32, img_size=(224, 224), is_training=True):
    # 1. Nạp dữ liệu
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size
    )

    # 2. Định nghĩa các lớp tiền xử lý
    # Chuẩn hóa pixel về [0, 1]
    rescale_layer = layers.Rescaling(1./255)
    
    # Tăng cường dữ liệu (Chỉ dùng khi training)
    augmentation_layers = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # 3. Áp dụng các kỹ thuật xử lý vào Pipeline
    # Luôn áp dụng chuẩn hóa
    ds = ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Chỉ áp dụng tăng cường nếu là tập Training
    if is_training:
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)

    # 4. Tối ưu hóa performance
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

if __name__ == "__main__":
    print("\n--- ĐANG CHẠY THỬ PIPELINE ---")
    try:
        # Thử nạp 1 batch với kích thước 1
        test_ds = get_pipeline('data', batch_size=1)
        
        # Lấy thử 1 phần tử để xem nó có hoạt động không
        for images, labels in test_ds.take(1):
            print("=> Pipeline hoạt động tốt!")
            print(f"=> Kích thước batch sau khi nạp: {images.shape}")
            print(f"=> Giá trị pixel tối đa: {tf.reduce_max(images):.2f}")
    except Exception as e:
        print(f"=> Pipeline bị lỗi: {e}")


