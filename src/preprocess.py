import tensorflow as tf

DATA_PATH = 'data/'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_and_split_data():
    # Bước 1: Lấy 80% cho Training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2, # Lấy 20% còn lại để lát nữa chia tiếp
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Bước 2: Lấy 20% còn lại làm tập "Tạm thời"
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Bước 3: Chia tập "Tạm thời" làm 2 nửa bằng nhau (10% Val, 10% Test)
    # Tính toán số lượng batch trong tập tạm thời
    batches = tf.data.experimental.cardinality(temp_ds)
    val_batches = batches // 2
    
    val_ds = temp_ds.take(val_batches)
    test_ds = temp_ds.skip(val_batches)

    # Bước 4: Chuẩn hóa (Normalization)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Tối ưu hiệu năng
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train, val, test = load_and_split_data()
    print(f"Số lượng batch tập Train: {tf.data.experimental.cardinality(train)}")
    print(f"Số lượng batch tập Val: {tf.data.experimental.cardinality(val)}")
    print(f"Số lượng batch tập Test: {tf.data.experimental.cardinality(test)}")