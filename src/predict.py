import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Đăng ký hỗ trợ đọc định dạng HEIC/HEIF (ảnh chụp mặc định của iPhone)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # nếu chưa cài pillow-heif, các định dạng thường (jpg/png/webp) vẫn hoạt động bình thường

# Dựng lại ĐÚNG kiến trúc như lúc train (base MobileNetV2 + GAP + Dense128 + Dropout + Dense6)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Load weights đã lưu (thay vì load cả model) -> tránh lỗi tương thích version Keras
model.load_weights("models/model_deploy_v2_finetuned_combined.weights.h5")

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# ⚠️ KHÔNG đổi thứ tự / nội dung class_names ở trên — nó phải khớp 100% với thứ tự
# lúc train (index 0-5 của output model gắn cứng với thứ tự này). Đây chỉ là "tên khóa"
# nội bộ, không phải nhãn hiển thị.

# Nhãn hiển thị song ngữ (Việt - Anh) — CHỈ dùng cho giao diện, không ảnh hưởng model.
LABELS_VI = {
    'cardboard': 'Bìa carton',
    'glass':     'Thủy tinh',
    'metal':     'Kim loại',
    'paper':     'Giấy',
    'plastic':   'Nhựa',
    'trash':     'Rác thải khác',
}

# ============================================================
# NHÓM RÁC LỚN (theo cách phân loại rác tại nguồn phổ biến ở VN)
# Dùng cho phần giáo dục tổng quan (sidebar) — độc lập với 6 class
# model nhận diện được. Màu thùng chỉ mang tính tham khảo chung,
# thực tế có thể khác nhau tùy địa phương/đơn vị thu gom.
# ============================================================
GROUP_INFO = {
    'huu_co': {
        'name': 'Rác hữu cơ',
        'icon': 'leaf',
        'color': '#2E8B57',
        'description': 'Rác dễ phân hủy sinh học: thức ăn thừa, rau củ quả, lá cây, bã trà/cà phê...',
        'bin_note': 'Thường được thu gom riêng để ủ phân compost, hạn chế đưa vào bãi chôn lấp.',
    },
    'vo_co_tai_che': {
        'name': 'Rác vô cơ — có thể tái chế',
        'icon': 'recycle',
        'color': '#4A90D9',
        'description': 'Vật liệu có thể xử lý và tái sử dụng: giấy, bìa carton, nhựa, kim loại, thủy tinh.',
        'bin_note': 'Nên để riêng, khô ráo, sạch sẽ để bán/thu gom phế liệu thay vì lẫn vào rác thường.',
    },
    'vo_co_khac': {
        'name': 'Rác vô cơ khác — không tái chế',
        'icon': 'trash-2',
        'color': '#B03A2E',
        'description': 'Rác khó hoặc không thể tái chế bằng quy trình thông thường (nhựa hỗn hợp, đồ sành sứ vỡ...).',
        'bin_note': 'Xử lý như rác thải sinh hoạt còn lại — chôn lấp hoặc đốt.',
    },
}

# ============================================================
# THÔNG TIN CHI TIẾT TỪNG LOẠI RÁC (6 class model nhận diện được)
# dùng để hiển thị giáo dục trên giao diện. Không ảnh hưởng gì
# tới model, chỉ là nội dung tĩnh nên có thể chỉnh sửa/bổ sung tự do.
# 'icon_lucide' là tên icon trong bộ Lucide (xem src/icons.py).
# ============================================================
WASTE_INFO = {
    'cardboard': {
        'icon_lucide': 'package',
        'color': '#C68642',
        'group': 'vo_co_tai_che',
        'description': 'Bìa carton, thùng giấy, hộp đóng gói làm từ giấy ép nhiều lớp.',
        'recyclable': True,
        'recycle_note': 'Tái chế được. Nên làm phẳng thùng và giữ khô ráo trước khi bỏ vào rác tái chế.',
        'examples': ['Thùng carton', 'Hộp giày', 'Lõi giấy vệ sinh', 'Bao bì đóng hàng'],
        'tips': 'Gỡ băng keo, ghim bấm trước khi tái chế. Carton dính dầu mỡ (VD: hộp pizza bẩn) nên bỏ vào rác thường.',
    },
    'glass': {
        'icon_lucide': 'glass-water',
        'color': '#2E8B57',
        'group': 'vo_co_tai_che',
        'description': 'Chai lọ, vật dụng làm từ thủy tinh.',
        'recyclable': True,
        'recycle_note': 'Tái chế được gần như vô hạn lần mà không giảm chất lượng. Nên rửa sạch trước khi bỏ.',
        'examples': ['Chai thủy tinh', 'Lọ mứt/lọ gia vị', 'Ly vỡ', 'Bóng đèn (cần xử lý riêng)'],
        'tips': 'Thủy tinh vỡ nên bọc lại để tránh gây thương tích cho người thu gom. Gương và bóng đèn KHÔNG cùng loại tái chế với chai lọ thường.',
    },
    'metal': {
        'icon_lucide': 'magnet',
        'color': '#8C8C8C',
        'group': 'vo_co_tai_che',
        'description': 'Lon, hộp kim loại, vật dụng làm từ nhôm hoặc thép.',
        'recyclable': True,
        'recycle_note': 'Tái chế được, giá trị tái chế cao. Nên súc rửa sơ trước khi bỏ.',
        'examples': ['Lon nước ngọt', 'Hộp thiếc đựng thực phẩm', 'Nắp chai kim loại', 'Vỏ lon sữa'],
        'tips': 'Có thể ép dẹp lon để tiết kiệm diện tích chứa. Kim loại dính sơn, hóa chất nên hỏi nơi thu gom trước khi bỏ chung.',
    },
    'paper': {
        'icon_lucide': 'file-text',
        'color': '#4A90D9',
        'group': 'vo_co_tai_che',
        'description': 'Giấy in, giấy báo, giấy viết và các sản phẩm giấy mỏng.',
        'recyclable': True,
        'recycle_note': 'Tái chế được, nhưng chỉ tái chế được giới hạn số lần (sợi giấy ngắn dần).',
        'examples': ['Báo cũ', 'Giấy in/photo', 'Sách vở cũ', 'Phong bì'],
        'tips': 'Giấy dính dầu mỡ, giấy ăn đã dùng, giấy bóng kính KHÔNG tái chế được — nên bỏ vào rác thường.',
    },
    'plastic': {
        'icon_lucide': 'shopping-bag',
        'color': '#E4B800',
        'group': 'vo_co_tai_che',
        'description': 'Chai, hộp, túi nhựa và các vật dụng làm từ nhựa.',
        'recyclable': True,
        'recycle_note': 'Tái chế được tùy loại nhựa (xem ký hiệu số 1-7 dưới đáy sản phẩm). Nên rửa sạch trước khi bỏ.',
        'examples': ['Chai nước suối', 'Hộp nhựa đựng thực phẩm', 'Túi nhựa cứng', 'Vỏ chai dầu gội'],
        'tips': 'Nhựa dùng một lần (ống hút, túi nilon mỏng) khó tái chế, nên hạn chế sử dụng. Ép dẹp chai trước khi bỏ để tiết kiệm không gian.',
    },
    'trash': {
        'icon_lucide': 'trash-2',
        'color': '#B03A2E',
        'group': 'vo_co_khac',
        'description': 'Rác không thuộc 5 nhóm trên — thường là rác hỗn hợp hoặc khó tái chế.',
        'recyclable': False,
        'recycle_note': 'Thường KHÔNG tái chế được bằng quy trình thông thường, xử lý như rác thải sinh hoạt.',
        'examples': ['Tã/băng vệ sinh', 'Đồ sành sứ vỡ', 'Vật dụng nhựa hỗn hợp nhiều lớp', 'Rác thực phẩm'],
        'tips': 'Nếu là rác thực phẩm, nên tách riêng làm rác hữu cơ để ủ phân thay vì chôn lấp — xem thêm ở mục "Nhóm rác" trên sidebar.',
    },
}


def predict_image(uploaded_file):
    # .convert("RGB") xử lý được mọi định dạng PIL đọc ra (kể cả ảnh có kênh alpha như PNG/WEBP,
    # hoặc ảnh grayscale) -> luôn ép về đúng 3 kênh màu mà model cần, tránh lỗi shape
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)[0]  # mảng xác suất của cả 6 class

    idx = int(np.argmax(pred))
    label_en = class_names[idx]
    label_vi = LABELS_VI[label_en]
    confidence = float(pred[idx]) * 100

    # Top-3 xác suất cao nhất, dùng để hiển thị biểu đồ trên giao diện
    top3_idx = np.argsort(pred)[::-1][:3]
    top3 = [
        {
            'label_en': class_names[i],
            'label_vi': LABELS_VI[class_names[i]],
            'confidence': float(pred[i]) * 100,
        }
        for i in top3_idx
    ]

    return label_en, label_vi, confidence, top3