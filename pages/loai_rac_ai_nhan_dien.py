import streamlit as st

from src.predict import class_names, LABELS_VI
from src.ui import render_waste_card

st.title("📦 6 loại rác AI có thể nhận diện")
st.write(
    "Model được huấn luyện để phân biệt 6 loại rác dưới đây. Bấm vào từng mục để "
    "xem mô tả, ví dụ vật dụng và cách xử lý phù hợp."
)

for class_key in class_names:
    with st.expander(f"{LABELS_VI[class_key]} ({class_key})"):
        render_waste_card(class_key)