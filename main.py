import tensorflow as tf
import os
from src.pipeline import get_pipeline
from src.model_builder_v2 import build_cnn_model

def train_model():
    print("=== BẮT ĐẦU TRAIN MODEL ===")

    # 1. Đường dẫn dữ liệu
    data_dir = 'data_final'
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    if not os.path.exists(data_dir):
        print(f"LỖI: Không tìm thấy thư mục {data_dir}")
        return

    # 2. Load dataset
    print("-> Đang load dữ liệu...")
    train_ds = get_pipeline(train_path, is_training=True)
    val_ds = get_pipeline(val_path, is_training=False)

    # 3. Build model
    print("-> Khởi tạo model...")
    model = build_cnn_model(input_shape=(224, 224, 3), num_classes=4)

    # 4. Callbacks 
    callbacks = [
        # Lưu model tốt nhất
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        
        # Dừng sớm nếu không cải thiện
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        
        # Giảm learning rate khi plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6
        )
    ]

    # 5. Train model
    print("-> Bắt đầu huấn luyện...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,   
        callbacks=callbacks
    )

    # 6. Lưu model cuối
    model.save("final_model.h5")

    print("=== TRAIN HOÀN TẤT ===")

if __name__ == "__main__":
    train_model()

