import streamlit as st
import pandas as pd
from src.predict import predict_image, LABELS_VI, WASTE_INFO, GROUP_INFO, class_names
from src.icons import icon_html, icon_title

st.set_page_config(page_title="Phân loại rác thải AI", page_icon="♻️", layout="wide")

st.title("♻️ Phân loại rác thải bằng AI")
st.write(
    "Tải lên ảnh rác thải để hệ thống phân loại tự động, đồng thời tìm hiểu cách "
    "xử lý và nên bỏ vào thùng nào cho phù hợp."
)

# ------------------------------------------------------------------
# SIDEBAR: giới thiệu 3 nhóm rác lớn (giáo dục tổng quan, không phụ
# thuộc vào model — mở rộng ra cả rác hữu cơ mà model không nhận diện)
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(icon_title("recycle", " Nhóm rác thải", size=24), unsafe_allow_html=True)
    st.caption(
        "Cách phân loại rác tại nguồn phổ biến ở Việt Nam gồm 3 nhóm chính. "
        "Màu thùng chỉ mang tính tham khảo, có thể khác nhau tùy địa phương."
    )
    for group_key, group in GROUP_INFO.items():
        st.markdown(
            f"""
            <div style="border-left:4px solid {group['color']}; padding:8px 12px; margin-bottom:10px; background:rgba(0,0,0,0.02); border-radius:4px;">
                {icon_title(group['icon'], ' <b>' + group['name'] + '</b>', size=20)}
                <div style="font-size:0.85em; margin-top:4px;">{group['description']}</div>
                <div style="font-size:0.8em; margin-top:4px; opacity:0.75;">🗑 {group['bin_note']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption(
        "Lưu ý: model hiện chỉ nhận diện 6 loại rác vô cơ (cardboard, glass, "
        "metal, paper, plastic, trash) — chưa nhận diện được rác hữu cơ."
    )

tab_classify, tab_info = st.tabs(["🔍 Phân loại ảnh", "📚 Tìm hiểu 6 loại rác"])


def render_waste_card(class_key: str):
    """Hiển thị thông tin chi tiết 1 loại rác dạng thẻ thông tin, dùng icon Lucide."""
    info = WASTE_INFO[class_key]
    group = GROUP_INFO[info["group"]]
    label = LABELS_VI[class_key]

    badge_icon = "recycle" if info["recyclable"] else "trash-2"
    badge_text = "Có thể tái chế" if info["recyclable"] else "Khó / không tái chế"

    st.markdown(icon_title(info["icon_lucide"], f" {label} ({class_key})", size=26, tag="h4"), unsafe_allow_html=True)
    st.markdown(icon_html(badge_icon, size=16) + f"<i>{badge_text}</i>", unsafe_allow_html=True)
    st.write(info["description"])
    st.markdown(f"**Ví dụ:** {', '.join(info['examples'])}")
    st.info(f"**Cách xử lý:** {info['recycle_note']}")
    st.markdown(f"💡 **Mẹo:** {info['tips']}")

    st.markdown(
        icon_title(group["icon"], f" Nên bỏ vào: <b>{group['name']}</b>", size=18) +
        f"<div style='font-size:0.85em; opacity:0.8; margin-top:2px;'>{group['bin_note']}</div>",
        unsafe_allow_html=True,
    )


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
                st.markdown(icon_title(info["icon_lucide"], f" Kết quả: {label_vi}", size=28, tag="h3"), unsafe_allow_html=True)
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
        info = WASTE_INFO[class_key]
        with st.expander(f"{LABELS_VI[class_key]} ({class_key})"):
            render_waste_card(class_key)