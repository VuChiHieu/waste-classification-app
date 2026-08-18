import streamlit as st
from src.ui import render_footer
from src.predict import GROUP_INFO, WASTE_INFO, LABELS_VI, class_names
from src.ui import render_group_card

st.title("🗂️ Nhóm rác thải — thông tin thêm")
st.write(
    "Ngoài 6 loại rác mà AI nhận diện trực tiếp, rác thải nói chung thường được chia "
    "thành 3 nhóm lớn theo cách phân loại rác tại nguồn phổ biến ở Việt Nam. Trang này "
    "giúp bạn hiểu bức tranh tổng quan hơn — kể cả những loại rác AI hiện **chưa** nhận "
    "diện được, ví dụ rác hữu cơ."
)
st.caption("Màu thùng / cách phân loại chỉ mang tính tham khảo chung, thực tế có thể khác nhau tùy địa phương.")

cols = st.columns(3)
for col, group_key in zip(cols, GROUP_INFO.keys()):
    with col:
        render_group_card(group_key)

st.divider()
st.subheader("6 loại rác AI nhận diện thuộc nhóm nào?")
for class_key in class_names:
    info = WASTE_INFO[class_key]
    group = GROUP_INFO[info["group"]]
    st.markdown(f"- **{LABELS_VI[class_key]}** → {group['name']}")
    
render_footer()