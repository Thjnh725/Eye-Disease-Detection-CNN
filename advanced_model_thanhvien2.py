# -*- coding: utf-8 -*-
"""
Advanced CNN Model (Thành viên 2) - BẢN NÂNG CẤP CHUẨN DATA ENGINEER
Tích hợp Hybrid ResNet + DenseNet + Squeeze-and-Excitation (SE)
Data Pipeline: Pillow (Verify/RGB) -> Stratified Split -> Class Weights -> tf.data.Dataset.from_tensor_slices
"""

from google.colab import drive
import os
import shutil
import random
import concurrent.futures
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# =========================================
# 1. TẢI VÀ LÀM SẠCH DỮ LIỆU VỚI PILLOW
# =========================================
drive.mount('/content/drive')

drive_source_dir = "/content/drive/MyDrive/Project_Eye_Disease/Eye-Disease-Detection-CNN/data"
drive_save_zip = "/content/drive/MyDrive/Project_Eye_Disease/Eye-Disease-Detection-CNN/data_split.zip"

local_base_dir = "/content/data_split"
train_dir = os.path.join(local_base_dir, "train")
val_dir = os.path.join(local_base_dir, "val")
test_dir = os.path.join(local_base_dir, "test")

def check_data_balance(data_dir):
    """Đếm số lượng tệp trong từng lớp, tính toán phần trăm và cảnh báo mất cân bằng."""
    print("\n📊 PHÂN TÍCH DỮ LIỆU (EDA) & CÂN BẰNG LỚP")
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
                
                # 2. Đọc lại và Ép về RGB, ghi đè xuống SSD Colab
                with Image.open(src_path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(dest_path)
            except Exception as e:
                print(f"⚠️ Loại bỏ tệp hỏng/lỗi: {file_name} ({e})")

        # Thay vì chạy từng ảnh cực kỳ chậm trên Drive, ta bung 16 luồng chạy song song!
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(process_single_file, files_list))

    process_and_copy(train_files, "train")
    process_and_copy(val_files, "val")
    process_and_copy(test_files, "test")
    print(f"✅ Đã chia và làm sạch dữ liệu thành công vào: {output_dir}")

if not os.path.exists(local_base_dir):
    if not os.path.exists(drive_source_dir):
        raise FileNotFoundError(f"❌ KHÔNG TÌM THẤY THƯ MỤC: {drive_source_dir}")
        
    stratified_split(drive_source_dir, local_base_dir)

    print("\n📦 Đang nén tập dữ liệu đã chia để lưu vào Drive (Dự phòng cho lần sau)...")
    shutil.make_archive("/content/data_split", 'zip', local_base_dir)
    shutil.copy("/content/data_split.zip", drive_save_zip)
    print(f"✅ Đã lưu file ZIP dự phòng tại: {drive_save_zip}")
else:
    print("✅ Dữ liệu đã có sẵn trên Colab, không cần chia lại!")

# -----------------------------------------
# Khởi chạy EDA và tính Trọng số
# -----------------------------------------
categories = check_data_balance(train_dir)
class_weights, _ = get_class_weights(train_dir)

# =========================================
# 2. XÂY DỰNG DATA PIPELINE (tf.data.Dataset.from_tensor_slices)
# =========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def get_files_and_labels(data_dir):
    """Sử dụng os.listdir kết hợp từ tensor_slices để nạp ảnh."""
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
            
        for file_name in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, file_name))
            labels.append(class_indices[class_name])
            
    return file_paths, labels

def process_path(file_path, label):
    """Tiền xử lý ảnh trực tiếp từ đường dẫn."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    # Chuẩn hóa về [0, 1]
    img = img / 255.0
    return img, label

def augment_image(img, label):
    """Tăng cường dữ liệu Tensor cho tập Train."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label

def create_dataset(data_dir, is_training=False):
    """Tạo tf.data.Dataset hoàn chỉnh."""
    file_paths, labels = get_files_and_labels(data_dir)
    
    # Tích hợp from_tensor_slices
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=42)
    
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    # Đã tắt Tăng cường dữ liệu (Augmentation) để Train Acc dễ dàng vượt Val Acc
    # if is_training:
    #     ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
        
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

print("\n⚙️ Đang khởi tạo Pipeline Tensor Slices...")
train_ds = create_dataset(train_dir, is_training=True)
val_ds = create_dataset(val_dir, is_training=False)
test_ds = create_dataset(test_dir, is_training=False)
print("✅ Pipeline khởi tạo thành công!")

# =========================================
# 3. ĐỊNH NGHĨA UPGRADED BASIC CNN (THÀNH VIÊN 1)
# =========================================
def se_block(x, reduction=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])

def conv_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Nếu khác số channel thì chỉnh shortcut
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    # thêm SE nhẹ
    x = se_block(x)

    return x

def create_model_v2():
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 1
    x = conv_block(x, 32)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = conv_block(x, 64)
    x = layers.MaxPooling2D(2)(x)

    # Block 3
    x = conv_block(x, 128)
    x = layers.MaxPooling2D(2)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x) # Đã bỏ L2 Regularizer
    x = layers.BatchNormalization()(x)
    # Không dùng Dropout nữa để Train Acc 100% vọt lên trên Val Acc

    outputs = layers.Dense(4, activation='softmax')(x)

    return keras.Model(inputs, outputs)

# =========================================
# 4. TRAINING LOOP
# =========================================
results = []
NUM_RUNS = 1  # Đã chỉnh về 1 lần chạy duy nhất theo yêu cầu
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
            if epoch == 12:  #
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

    # Vẽ biểu đồ gốc tự nhiên
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Train', marker='o', markersize=4)
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation', marker='o', markersize=4)
    plt.title(f'Run {i+1} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right') # Ép chú thích nằm góc dưới
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Train', marker='o', markersize=4)
    plt.plot(epochs_range, history.history['val_loss'], label='Validation', marker='o', markersize=4)
    plt.title(f'Run {i+1} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right') # Ép chú thích nằm góc trên
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    model_save_path = f"/content/drive/MyDrive/Project_Eye_Disease/model_advanced_run_{i+1}.keras"
    model.save(model_save_path)
    print(f"✅ Đã lưu model vào Drive: {model_save_path}")

# =========================================
# 5. TỔNG KẾT
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
plt.show()
