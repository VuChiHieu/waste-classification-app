import streamlit as st
import pandas as pd
from src.predict import predict_image, LABELS_VI, WASTE_INFO, class_names

st.set_page_config(page_title="Phân loại rác thải AI", page_icon="♻️", layout="wide")

st.title("♻️ Phân loại rác thải bằng AI")
st.write(
    "Tải lên ảnh rác thải để hệ thống phân loại tự động, đồng thời tìm hiểu cách "
    "xử lý phù hợp cho từng loại rác."
)

tab_classify, tab_info = st.tabs(["🔍 Phân loại ảnh", "📚 Tìm hiểu các loại rác"])


def render_waste_card(class_key: str):
    """Hiển thị thông tin chi tiết 1 loại rác dạng thẻ thông tin."""
    info = WASTE_INFO[class_key]
    label = LABELS_VI[class_key]

    recyclable_badge = "✅ Có thể tái chế" if info["recyclable"] else "⚠️ Khó / không tái chế"

    st.markdown(f"#### {info['icon']} {label} ({class_key})")
    st.caption(recyclable_badge)
    st.write(info["description"])
    st.markdown(f"**Ví dụ:** {', '.join(info['examples'])}")
    st.info(f"**Cách xử lý:** {info['recycle_note']}")
    st.markdown(f"💡 **Mẹo:** {info['tips']}")


# ------------------------------------------------------------------
# TAB 1: Phân loại ảnh
# ------------------------------------------------------------------
with tab_classify:
    uploaded_file = st.file_uploader(
        "Chọn ảnh rác thải cần phân loại",
        type=["jpg", "jpeg", "png", "webp", "bmp", "heic", "heif"],
    )

    if uploaded_file:
        try:
            with st.spinner("Đang phân loại..."):
                label_en, label_vi, confidence, top3 = predict_image(uploaded_file)

            col_img, col_result = st.columns([1, 1.2], gap="large")

            with col_img:
                st.image(uploaded_file, use_container_width=True)

            with col_result:
                info = WASTE_INFO[label_en]
                st.markdown(f"### {info['icon']} Kết quả: {label_vi}")
                st.metric("Độ tin cậy", f"{confidence:.1f}%")

                # Biểu đồ top-3 xác suất
                st.markdown("**Top 3 dự đoán:**")
                df_top3 = pd.DataFrame(
                    {t["label_vi"]: t["confidence"] for t in top3}.items(),
                    columns=["Loại rác", "Xác suất (%)"],
                ).set_index("Loại rác")
                st.bar_chart(df_top3, horizontal=True)

                # Cảnh báo nhẹ nếu model không chắc chắn
                if confidence < 60:
                    st.warning(
                        "Độ tin cậy khá thấp — ảnh có thể mập mờ giữa nhiều loại rác. "
                        "Hãy thử chụp ảnh rõ nét hơn, sát vật thể hơn."
                    )

            st.divider()
            st.subheader("Thông tin & cách xử lý")
            render_waste_card(label_en)

        except Exception as e:
            st.error(f"Không thể xử lý ảnh này. Vui lòng thử ảnh khác (định dạng JPG/PNG). Chi tiết lỗi: {e}")
    else:
        st.info("👆 Tải ảnh lên để bắt đầu phân loại.")

# ------------------------------------------------------------------
# TAB 2: Thông tin 6 loại rác (đọc trước / tra cứu độc lập)
# ------------------------------------------------------------------
with tab_info:
    st.write("Tìm hiểu đặc điểm và cách xử lý của từng loại rác trong hệ thống phân loại.")
    for class_key in class_names:
        with st.expander(f"{WASTE_INFO[class_key]['icon']}  {LABELS_VI[class_key]}"):
            render_waste_card(class_key)