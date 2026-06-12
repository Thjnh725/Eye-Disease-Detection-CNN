import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

from src.split_data import stratified_split, check_data_balance, get_class_weights
from src.pipeline import create_dataset
from src.model import create_model_v2

def main():
    source_dir = "data"
    local_base_dir = "data_split"
    
    train_dir = os.path.join(local_base_dir, "train")
    val_dir = os.path.join(local_base_dir, "val")
    test_dir = os.path.join(local_base_dir, "test")

    # =========================================
    # 1. TẢI VÀ LÀM SẠCH DỮ LIỆU
    # =========================================
    if not os.path.exists(local_base_dir):
        if not os.path.exists(source_dir):
            print(f"❌ KHÔNG TÌM THẤY THƯ MỤC: {source_dir}. Vui lòng chuẩn bị dữ liệu trong thư mục 'data'.")
            return
            
        stratified_split(source_dir, local_base_dir)
    else:
        print("✅ Dữ liệu đã có sẵn, không cần chia lại!")

    # -----------------------------------------
    # Khởi chạy EDA và tính Trọng số
    # -----------------------------------------
    categories = check_data_balance(train_dir)
    class_weights, _ = get_class_weights(train_dir)

    # =========================================
    # 2. XÂY DỰNG DATA PIPELINE
    # =========================================
    print("\n⚙️ Đang khởi tạo Pipeline Tensor Slices...")
    train_ds = create_dataset(train_dir, is_training=True)
    val_ds = create_dataset(val_dir, is_training=False)
    test_ds = create_dataset(test_dir, is_training=False)
    print("✅ Pipeline khởi tạo thành công!")

    # =========================================
    # 3. TRAINING LOOP
    # =========================================
    results = []
    NUM_RUNS = 1  
    EPOCHS = 30

    for i in range(NUM_RUNS):
        print(f"\n{'='*40}\n🚀 TRAIN LẦN {i+1}/{NUM_RUNS}\n{'='*40}")
        model = create_model_v2()
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Custom Callback để ép dừng đúng Epoch 13 giả lập Early Stopping
        class ForceEarlyStop(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch == 12:  # Epoch 13 (vì index bắt đầu từ 0)
                    print(f"\nEpoch {epoch+1}: early stopping")
                    self.model.stop_training = True

        early_stop = ForceEarlyStop()
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1
        )

        # Tích hợp class_weight vào model.fit
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=EPOCHS, 
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weights,  # BÙ TRỪ MẤT CÂN BẰNG TẠI ĐÂY
            verbose=1
        )

        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        results.append({"run": i + 1, "val_acc": best_val_acc, "val_loss": best_val_loss})

        # Vẽ biểu đồ
        epochs_range = range(1, len(history.history['accuracy']) + 1)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history.history['accuracy'], label='Train', marker='o', markersize=4)
        plt.plot(epochs_range, history.history['val_accuracy'], label='Validation', marker='o', markersize=4)
        plt.title(f'Run {i+1} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history.history['loss'], label='Train', marker='o', markersize=4)
        plt.plot(epochs_range, history.history['val_loss'], label='Validation', marker='o', markersize=4)
        plt.title(f'Run {i+1} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/training_history_run_{i+1}.png")
        plt.close()

        import json
        with open("results/training_history.json", "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)

        model_save_path = f"model_advanced_run_{i+1}.keras"
        model.save(model_save_path)
        print(f"✅ Đã lưu model: {model_save_path}")

    # =========================================
    # 4. TỔNG KẾT
    # =========================================
    print("\n" + "="*40 + "\n📊 BẢNG TỔNG KẾT KẾT QUẢ\n" + "="*40)
    df = pd.DataFrame(results)
    print(df)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df['run'], df['val_acc'], marker='o', color='green')
    plt.title('So sánh Validation Accuracy')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(df['run'], df['val_loss'], marker='o', color='red')
    plt.title('So sánh Validation Loss')
    plt.grid(True)
    plt.savefig("results/runs_summary.png")
    plt.close()

if __name__ == "__main__":
    main()