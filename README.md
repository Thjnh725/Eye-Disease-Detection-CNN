# Eye Disease Detection – Advanced CNN 🔬

Hệ thống phân loại bệnh mắt dựa trên ảnh võng mạc, sử dụng CNN với các thành phần Residual và SE Attention để cải thiện chất lượng dự đoán.

---

## 📋 Mục tiêu

Phân loại 4 nhóm ảnh võng mạc:

| Nhãn | Bệnh |
|------|------|
| `cataract` | Đục thủy tinh thể |
| `diabetic_retinopathy` | Bệnh võng mạc tiểu đường |
| `glaucoma` | Bệnh tăng nhãn áp |
| `normal` | Mắt bình thường |

---

## 📁 Cấu trúc Dự án

```
Eye-Disease-Detection-CNN/
├── data/                 # Dữ liệu thô chưa chia
│   ├── cataract/
│   ├── diabetic_retinopathy/
│   ├── glaucoma/
│   └── normal/
├── data_split/           # Dữ liệu đã chia train/val/test
│   ├── train/
│   ├── val/
│   └── test/
├── results/              # Kết quả training và đồ thị
├── logs/                 # TensorBoard logs
├── src/
│   ├── model.py          # Định nghĩa kiến trúc CNN
│   ├── pipeline.py       # Data pipeline với tf.data
│   └── split_data.py     # Chia dữ liệu, tính weights, kiểm tra balance
├── main.py               # Entry point để chia dữ liệu và huấn luyện
├── requirements.txt      # Dependencies
└── README.md             # Tài liệu dự án
```

---

## 🧠 Kiến trúc mô hình chính

Mô hình `create_model_v2()` trong `src/model.py` có cấu trúc:

- Input: `224×224×3`
- Conv2D + BatchNormalization + ReLU
- 3 residual block với `Conv2D`, `BatchNormalization`, `Add` skip connection
- Mỗi residual block tích hợp `SE attention`
- MaxPooling giảm kích thước không gian
- GlobalAveragePooling2D
- Dense(128) + BatchNormalization
- Output Dense(4, softmax)

### Các tính năng chính

- Residual connection giúp ổn định huấn luyện
- SE block cải thiện khả năng chú ý channel
- BatchNormalization giúp mô hình hội tụ nhanh hơn
- Class weights cân bằng học khi dữ liệu mất cân bằng

---

## ⚡ Cài đặt và chạy

### 1. Thiết lập môi trường

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt ảnh vào thư mục `data/` theo cấu trúc lớp:

```
data/
  cataract/
  diabetic_retinopathy/
  glaucoma/
  normal/
```

### 3. Chạy huấn luyện

```bash
python main.py
```

`main.py` thực hiện:

1. Chia dữ liệu từ `data/` sang `data_split/` nếu chưa tồn tại
2. Tạo pipeline `tf.data.Dataset`
3. Huấn luyện mô hình với `class_weight`
4. Lưu `model_advanced_run_<n>.keras`
5. Xuất kết quả đồ thị vào `results/`

### 4. Kết quả đầu ra

Sau khi chạy, một số file có thể được tạo:

- `results/training_history_run_1.png`
- `results/runs_summary.png`
- `results/training_history.json`
- `model_advanced_run_1.keras`

---

## 📌 Lưu ý

- Nếu thư mục `data_split/` đã tồn tại, `main.py` sẽ không chia lại dữ liệu.
- `src/pipeline.py` hiện tại chỉ chuẩn hóa ảnh và bật augmentation cơ bản.
- `src/split_data.py` thực hiện stratified split 80/10/10.

---

## 📦 Dependencies chính

- TensorFlow
- NumPy
- Pillow
- scikit-learn
- matplotlib
- pandas

---

## 🧪 Thông tin thêm

- `src/split_data.py` in ra số lượng ảnh mỗi lớp và cảnh báo mất cân bằng
- `src/pipeline.py` tạo `tf.data.Dataset` và có thể mở rộng augmentation
- `src/model.py` định nghĩa mô hình CNN với SE block

Nếu cần mở rộng thêm `predict` hoặc `Grad-CAM`, có thể bổ sung script riêng cho inference và giải thích mô hình.
