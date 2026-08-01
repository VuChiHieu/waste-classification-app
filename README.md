# ♻️ Waste Classification App

Ứng dụng phân loại rác thải tự động bằng Deep Learning (Transfer Learning với MobileNetV2), triển khai qua Streamlit.

**🔗 Demo trực tiếp:** [waste-classification-app.streamlit.app](https://waste-classification-app.streamlit.app) *(cập nhật lại link thật của bạn)*

## Giới thiệu

Dự án xây dựng hệ thống nhận diện và phân loại rác thải qua ảnh chụp, hỗ trợ 6 loại rác phổ biến:

| Class | Mô tả |
|---|---|
| `cardboard` | Bìa carton |
| `glass` | Thủy tinh |
| `metal` | Kim loại |
| `paper` | Giấy |
| `plastic` | Nhựa |
| `trash` | Rác không tái chế được |

Đồ án môn Thực tập tốt nghiệp, ngành Công nghệ thông tin.

## Kết quả mô hình

Model cuối cùng (MobileNetV2, fine-tuned trên dataset gộp từ TrashNet + Garbage Classification):

| Chỉ số | Giá trị |
|---|---|
| Test Accuracy | **89%** |
| Dataset | 6437 ảnh (TrashNet + Garbage Classification 12-class, đã lọc trùng) |
| F1-score thấp nhất (plastic) | 0.79 |
| F1-score cao nhất (cardboard) | 0.93 |

Xem chi tiết toàn bộ quá trình thực nghiệm (baseline → fine-tune → gộp dữ liệu → fine-tune lại) trong file `huong-dan-full-pipeline.md`.

## Cấu trúc project

```
waste-classification-app/
├── app.py                          # Giao diện Streamlit chính
├── requirements.txt                 # Thư viện cần cài
├── models/
│   └── model_deploy_final.weights.h5   # Trọng số model đã fine-tune
├── src/
│   ├── __init__.py
│   └── predict.py                   # Logic dựng model + dự đoán ảnh
└── README.md
```

## Chạy thử ở máy local

```bash
git clone https://github.com/TEN_TAI_KHOAN/waste-classification-app.git
cd waste-classification-app
pip install -r requirements.txt
streamlit run app.py
```

## Công nghệ sử dụng

- **Model:** MobileNetV2 (Transfer Learning, fine-tuned)
- **Framework:** TensorFlow / Keras, Streamlit
- **Huấn luyện:** Google Colab (GPU T4)
- **Dataset:** [TrashNet](https://github.com/garythung/trashnet), [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Deploy:** Streamlit Community Cloud

## Ghi chú kỹ thuật

- Model được lưu dưới dạng **weights** (`.weights.h5`) thay vì lưu cả model (`.h5`) để tránh lỗi tương thích phiên bản Keras giữa môi trường train (Colab) và môi trường deploy (Streamlit Cloud). Kiến trúc model được dựng lại bằng code trong `src/predict.py` trước khi load weights vào.
- App hỗ trợ nhiều định dạng ảnh phổ biến (JPG, PNG, WEBP, HEIC...) nhờ `Pillow` kết hợp `pillow-heif`.

## Hướng phát triển

- Mở rộng thêm class (battery, biological, clothes, shoes...)
- Thêm tính năng gợi ý cách xử lý/tái chế sau khi phân loại
- Cải thiện khả năng phân biệt `plastic` và `glass` (2 class dễ nhầm lẫn nhất hiện tại)

---
*Đồ án thực tập tốt nghiệp — [Tên bạn] — [Trường/Khoa] — 2026*
