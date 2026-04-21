import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    model = models.Sequential([
        # Khối 1: Trích xuất đặc trưng (Sử dụng SeparableConv2D để tối ưu)
        layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Khối 2
        layers.SeparableConv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Khối 3
        layers.SeparableConv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Tối ưu hóa bằng GAP thay vì Flatten
        layers.GlobalAveragePooling2D(), 
        
        # Dense layer 
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model