# ♻️ Waste Classification App

Ứng dụng phân loại rác thải tự động bằng Deep Learning (Transfer Learning với MobileNetV2), triển khai qua Streamlit.

**🔗 Demo trực tiếp:** [waste-classification-app.streamlit.app](https://waste-classification-app-3mte5ygc3mwbzkpazxyulo.streamlit.app/)
## Giới thiệu

Dự án xây dựng hệ thống nhận diện và phân loại rác thải qua ảnh chụp, hỗ trợ 6 loại rác phổ biến, kèm thông tin hướng dẫn xử lý/tái chế cho từng loại:

| Class | Mô tả |
|---|---|
| `cardboard` | Bìa carton |
| `glass` | Thủy tinh |
| `metal` | Kim loại |
| `paper` | Giấy |
| `plastic` | Nhựa |
| `trash` | Rác không tái chế được |

Ngoài dự đoán, app còn cung cấp trang giáo dục về 3 nhóm rác lớn (hữu cơ, vô cơ tái chế, vô cơ khác) để mở rộng bối cảnh ra ngoài 6 class model nhận diện được.

Đồ án môn Thực tập tốt nghiệp, ngành Công nghệ thông tin.

## Kết quả mô hình

Model cuối cùng (MobileNetV2, fine-tuned trên dataset gộp từ TrashNet + Garbage Classification):

| Chỉ số | Giá trị |
|---|---|
| Test Accuracy | **91%** |
| Dataset | 6437 ảnh (TrashNet + Garbage Classification 12-class, đã lọc trùng) |
| F1-score thấp nhất (plastic) | 0.84 |
| F1-score cao nhất (cardboard) | 0.94 |

Xem chi tiết toàn bộ quá trình thực nghiệm (baseline → fine-tune → gộp dữ liệu → fine-tune lại) trong [`docs/pipeline.md`](docs/pipeline.md), hoặc notebook gốc trong [`notebooks/`](notebooks/).

## Cấu trúc project

```
waste-classification-app/
├── app.py                          # Điều hướng đa trang (st.navigation)
├── requirements.txt                # Thư viện cần cài
├── runtime.txt                     # Phiên bản Python cho Streamlit Cloud
├── models/
│   └── model_deploy_final.weights.h5   # Trọng số model đã fine-tune
├── pages/
│   ├── ai_phan_loai_rac.py         # Trang chính: upload ảnh, dự đoán, top-3
│   ├── loai_rac_ai_nhan_dien.py    # Giới thiệu 6 loại rác model nhận diện
│   └── nhom_rac_thai.py            # Giáo dục: 3 nhóm rác lớn
├── src/
│   ├── __init__.py
│   ├── predict.py                  # Logic dựng model + dự đoán ảnh
│   ├── ui.py                       # Component giao diện dùng chung
│   └── icons.py                    # Helper hiển thị icon Lucide
├── notebooks/                      # Notebook Colab: EDA, tiền xử lý, train, fine-tune
├── docs/
│   └── pipeline.md                 # Tài liệu chi tiết toàn bộ pipeline
└── README.md
```

## Chạy thử ở máy local

```bash
git clone https://github.com/VuChiHieu/waste-classification-app
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
- Nhãn kết quả hiển thị **song ngữ Việt–Anh**, tách biệt hoàn toàn khỏi tên class nội bộ dùng cho model (đảm bảo không ảnh hưởng tới độ chính xác dự đoán).

## Hướng phát triển

- Mở rộng thêm class (battery, biological, clothes, shoes...)
- Cải thiện khả năng phân biệt `plastic` — class hiện có độ chính xác thấp nhất, dễ nhầm với `metal`/`glass`
- Thêm chức năng lưu lịch sử phân loại của người dùng

---
*Đồ án thực tập tốt nghiệp — Vũ Chí Hiếu/Phạm Nhật Huy — [ĐH Giao Thông Vận Tải/Khoa CNTT] — 2026*