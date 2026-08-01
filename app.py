import streamlit as st
from src.predict import predict_image

st.set_page_config(page_title="Phân loại rác thải AI", page_icon="♻️")

st.title("♻️ Phân loại rác thải bằng AI")
st.write("Tải lên ảnh rác thải để hệ thống phân loại tự động (6 loại: cardboard, glass, metal, paper, plastic, trash)")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, width=300)
    with st.spinner("Đang phân loại..."):
        label, confidence = predict_image(uploaded_file)
    st.success(f"Kết quả: **{label}** (độ tin cậy {confidence:.2f}%)")
