import streamlit as st
import pandas as pd

from src.predict import predict_image, WASTE_INFO
from src.icons import icon_title
from src.ui import render_waste_card, render_eco_note, render_footer

st.title("♻️AI phân loại rác")
st.write(
    "Tải lên ảnh rác thải để hệ thống phân loại tự động, đồng thời xem cách xử lý "
    "và nên bỏ vào nhóm rác nào cho phù hợp."
)

uploaded_file = st.file_uploader(
    "Chọn ảnh rác thải cần phân loại",
    type=["jpg", "jpeg", "png", "webp", "bmp", "heic", "heif"],
)

if uploaded_file:
    try:
        with st.spinner("Đang phân loại..."):
            label_en, label_vi, confidence, top3, eco_tip = predict_image(uploaded_file)

        # Đếm số lần phân loại trong phiên làm việc (gamification nhẹ)
        st.session_state.setdefault("classify_count", 0)
        st.session_state["classify_count"] += 1

        col_img, col_result = st.columns([1, 1.2], gap="large")

        with col_img:
            st.image(uploaded_file, use_container_width=True)

        with col_result:
            info = WASTE_INFO[label_en]
            st.markdown(
                icon_title(info["icon_lucide"], f" Kết quả: {label_vi}", size=28, tag="h3"),
                unsafe_allow_html=True,
            )
            st.metric("Độ tin cậy", f"{confidence:.1f}%")

            st.markdown("**Top 3 dự đoán:**")
            df_top3 = pd.DataFrame(
                {t["label_vi"]: t["confidence"] for t in top3}.items(),
                columns=["Loại rác", "Xác suất (%)"],
            ).set_index("Loại rác")
            st.bar_chart(df_top3, horizontal=True)

            if confidence < 60:
                st.warning(
                    "Độ tin cậy khá thấp — ảnh có thể mập mờ giữa nhiều loại rác. "
                    "Hãy thử chụp ảnh rõ nét hơn, sát vật thể hơn."
                )

        # Eco note ngay dưới kết quả, trước khi vào phần thông tin chi tiết
        render_eco_note(label_en, eco_tip)
        st.caption(f"🌍 Đây là lần phân loại thứ **{st.session_state['classify_count']}** của bạn trong phiên này.")

        st.divider()
        st.subheader("Thông tin & cách xử lý")
        render_waste_card(label_en)

    except Exception as e:
        st.error(f"Không thể xử lý ảnh này. Vui lòng thử ảnh khác (định dạng JPG/PNG). Chi tiết lỗi: {e}")
else:
    st.info("👆 Tải ảnh lên để bắt đầu phân loại.")
    st.caption(
        "Muốn tìm hiểu trước? Xem trang **📦 6 loại rác AI nhận diện** hoặc "
        "**🗂️ Nhóm rác thải** ở sidebar bên trái."
    )

render_footer()