# Pipeline: Phân loại rác thải bằng Deep Learning (MobileNetV2)

Tài liệu mô tả toàn bộ quy trình xây dựng model phân loại 6 loại rác thải
(cardboard, glass, metal, paper, plastic, trash) và triển khai thành ứng dụng
web bằng Streamlit.

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Dữ liệu](#dữ-liệu)
3. [Tiền xử lý & Augmentation](#tiền-xử-lý--augmentation)
4. [Kiến trúc model](#kiến-trúc-model)
5. [Huấn luyện](#huấn-luyện)
6. [Kết quả](#kết-quả)
7. [Triển khai (Deployment)](#triển-khai-deployment)
8. [Cấu trúc ứng dụng Streamlit](#cấu-trúc-ứng-dụng-streamlit)

---

## Tổng quan

- **Bài toán:** phân loại ảnh rác thải thành 6 nhóm: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- **Kiến trúc:** Transfer Learning với MobileNetV2 (pretrained ImageNet) làm backbone.
- **Công cụ:** Google Colab (train), Google Drive (lưu trữ), GitHub + Streamlit Community Cloud (deploy).
- **Kết quả cuối cùng:** **91% test accuracy**, F1-score mọi class đều ≥ 0.84.

## Dữ liệu

Model được train trên bộ dữ liệu **gộp từ 2 nguồn** để khắc phục tình trạng mất cân bằng
và thiếu dữ liệu của class `trash` trong dataset gốc:

| Nguồn | Vai trò |
|---|---|
| [TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet) (2527 ảnh) | Dataset gốc, ảnh chụp studio, nền đồng nhất |
| [Garbage Classification (12 class)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | Lọc lấy 6 class khớp TrashNet (gộp `brown/green/white-glass` → `glass`), bổ sung thêm dữ liệu — đặc biệt cho `trash` |

Sau khi gộp và loại trùng lặp bằng hash MD5 nội dung ảnh, dataset cuối cùng có **6437 ảnh**,
chia theo tỷ lệ **70% train / 15% validation / 15% test** (`split-folders`, `seed=42`):

| Class | Tổng | Train | Val | Test |
|---|---|---|---|---|
| cardboard | 891 | 623 | 133 | 135 |
| glass | 2024 | 1416 | 303 | 305 |
| metal | 769 | 538 | 115 | 116 |
| paper | 1050 | 735 | 157 | 158 |
| plastic | 869 | 608 | 130 | 131 |
| trash | 834 | 583 | 125 | 126 |
| **Tổng** | **6437** | **4503** | **963** | **971** |

## Tiền xử lý & Augmentation

- Resize toàn bộ ảnh về `224×224` (kích thước input chuẩn của MobileNetV2), chuẩn hóa pixel về `[0,1]`.
- **Chỉ augment tập train** (val/test giữ nguyên để đánh giá công bằng trên dữ liệu thật):
  - Xoay ngẫu nhiên ±30°, dịch ngang/dọc ±20%, shear 15%, zoom ±20%, lật ngang.
- **Class weight** (`sklearn.compute_class_weight`, chế độ `balanced`) được áp dụng khi train, để bù lại sự chênh lệch số lượng ảnh giữa các class (`glass` có 2024 ảnh trong khi `metal` chỉ 769).

## Kiến trúc model

```python
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
```

## Huấn luyện

Quy trình train gồm 2 giai đoạn:

**1. Baseline** — đóng băng toàn bộ MobileNetV2 (`trainable=False`), chỉ train phần đầu
(Dense128 + Dropout + Dense6). Optimizer Adam, `learning_rate=0.001`.

**2. Fine-tuning** — mở khóa (`trainable=True`) các layer cuối cùng của MobileNetV2
(từ layer thứ 100 trở đi), train tiếp với `learning_rate=0.00001` (thấp hơn nhiều để
không phá vỡ trọng số pretrained).

Cả 2 giai đoạn dùng chung callback:

```python
EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True)
```

> Lưu ý kỹ thuật: 2 callback trên **luôn phải theo dõi cùng 1 chỉ số** (`val_accuracy`).
> Nếu để lệch nhau (ví dụ `EarlyStopping` theo `val_loss` còn `ModelCheckpoint` theo
> `val_accuracy`), model tốt nhất trong bộ nhớ và model tốt nhất lưu ra file có thể là
> 2 epoch khác nhau — nên luôn đánh giá bằng cách `load_model()` lại từ file checkpoint,
> không dùng thẳng biến model ngay sau khi `fit()`.

## Kết quả

So sánh các lần thực nghiệm trên tập test:

| Model | Dữ liệu | Test Accuracy | trash F1 |
|---|---|---|---|
| Baseline | TrashNet (2527 ảnh) | 82% | 0.63 |
| Fine-tuned (lr=1e-5, unfreeze 55 layer) | TrashNet | 81% | 0.68 |
| Baseline (Combined) | TrashNet + Garbage Classification (6437 ảnh) | 90% | 0.90 |
| **Fine-tuned (Combined)** | TrashNet + Garbage Classification | **91%** | **0.93** |

Classification report của model cuối cùng (`mobilenetv2_finetuned_combined`):

```
              precision    recall  f1-score   support

   cardboard       0.94      0.94      0.94       135
       glass       0.92      0.92      0.92       305
       metal       0.85      0.91      0.88       116
       paper       0.96      0.94      0.95       158
     plastic       0.83      0.84      0.84       131
       trash       0.96      0.91      0.93       126

    accuracy                           0.91       971
   macro avg       0.91      0.91      0.91       971
weighted avg       0.91      0.91      0.91       971
```

**Nhận xét chính:**
- Việc tăng số lượng và đa dạng dữ liệu (đặc biệt cho `trash`, từ 137 lên 834 ảnh gốc) mang lại cải thiện lớn hơn nhiều so với việc chỉ tinh chỉnh hyperparameter trên dataset nhỏ.
- Fine-tuning chỉ thực sự phát huy tác dụng khi đã có đủ dữ liệu nền tảng — trên dataset nhỏ ban đầu, fine-tuning từng gây overfitting và làm giảm nhẹ độ chính xác.

## Triển khai (Deployment)

1. **Lưu weights thay vì lưu cả model** (`model.save_weights(...)`, tên file kết thúc bằng
   `.weights.h5`) — tránh lỗi bất tương thích định dạng Keras giữa môi trường train (Colab)
   và môi trường deploy (Streamlit Cloud).
2. Trong `src/predict.py`, dựng lại đúng kiến trúc model bằng code, rồi `model.load_weights(...)`.
3. Đẩy code lên GitHub, kết nối repo với [Streamlit Community Cloud](https://streamlit.io/cloud) —
   mỗi lần `git push` app sẽ tự động build lại.

## Cấu trúc ứng dụng Streamlit

```
waste-classification-app/
├── app.py                          # Điều hướng đa trang (st.navigation)
├── models/
│   └── model_deploy_*.weights.h5   # Trọng số model đã train
├── pages/
│   ├── ai_phan_loai_rac.py         # Trang chính: upload ảnh, dự đoán, top-3
│   ├── loai_rac_ai_nhan_dien.py    # Giới thiệu 6 loại rác model nhận diện
│   └── nhom_rac_thai.py            # Giáo dục: 3 nhóm rác lớn (kể cả rác hữu cơ)
└── src/
    ├── predict.py                  # Model + dự đoán + dữ liệu mô tả từng loại rác
    ├── ui.py                       # Component giao diện dùng chung
    └── icons.py                    # Helper hiển thị icon Lucide
```

**Điểm đáng chú ý:**
- Kết quả dự đoán hiển thị **song ngữ Việt–Anh** (`LABELS_VI` trong `predict.py`), tách biệt hoàn toàn với `class_names` (tiếng Anh, thứ tự cố định khớp với index output của model).
- Ngoài nhãn dự đoán, ứng dụng còn cung cấp thông tin giáo dục cho từng loại rác (`WASTE_INFO`): mô tả, ví dụ, khả năng tái chế, mẹo xử lý — và phân nhóm rác lớn hơn (`GROUP_INFO`) để mở rộng bối cảnh ra ngoài 6 class model nhận diện được (ví dụ rác hữu cơ).
- Hỗ trợ upload ảnh định dạng HEIC/HEIF (mặc định của iPhone) qua thư viện `pillow-heif`.
