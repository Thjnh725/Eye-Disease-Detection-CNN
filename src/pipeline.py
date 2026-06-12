import os
import tensorflow as tf

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