import streamlit as st

st.set_page_config(page_title="Phân loại rác thải AI", page_icon="♻️", layout="wide")

# Định nghĩa từng trang tường minh (title/icon tùy chỉnh, không phụ thuộc tên file)
# -> sau này muốn thêm trang mới: tạo file trong pages/, rồi thêm 1 dòng st.Page(...) ở đây.
classify_page = st.Page(
    "pages/ai_phan_loai_rac.py", title="AI phân loại rác", icon="🔍", default=True
)
waste_types_page = st.Page(
    "pages/loai_rac_ai_nhan_dien.py", title="6 loại rác AI nhận diện", icon="📦"
)
waste_groups_page = st.Page(
    "pages/nhom_rac_thai.py", title="Nhóm rác thải (thông tin thêm)", icon="🗂️"
)

pg = st.navigation([classify_page, waste_types_page, waste_groups_page])
pg.run()