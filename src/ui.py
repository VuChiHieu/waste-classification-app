"""
Các thành phần giao diện dùng chung giữa nhiều trang (pages/).
Tách riêng ra đây để không phải copy-paste code khi thêm trang mới.
"""

import streamlit as st

from src.predict import WASTE_INFO, GROUP_INFO
from src.icons import icon_html, icon_title


def render_waste_card(class_key: str):
    """Hiển thị thẻ thông tin chi tiết 1 loại rác: mô tả, ví dụ, cách xử lý, nên bỏ vào nhóm nào."""
    from src.predict import LABELS_VI  # import cục bộ để tránh vòng lặp import nếu module mở rộng sau này

    info = WASTE_INFO[class_key]
    group = GROUP_INFO[info["group"]]
    label = LABELS_VI[class_key]

    badge_icon = "recycle" if info["recyclable"] else "trash-2"
    badge_text = "Có thể tái chế" if info["recyclable"] else "Khó / không tái chế"

    st.markdown(
        icon_title(info["icon_lucide"], f" {label} ({class_key})", size=26, tag="h4"),
        unsafe_allow_html=True,
    )
    st.markdown(icon_html(badge_icon, size=16) + f"<i>{badge_text}</i>", unsafe_allow_html=True)
    st.write(info["description"])
    st.markdown(f"**Ví dụ:** {', '.join(info['examples'])}")
    st.info(f"**Cách xử lý:** {info['recycle_note']}")
    st.markdown(f"💡 **Mẹo:** {info['tips']}")

    st.markdown(
        icon_title(group["icon"], f" Nên bỏ vào: <b>{group['name']}</b>", size=18)
        + f"<div style='font-size:0.85em; opacity:0.8; margin-top:2px;'>{group['bin_note']}</div>",
        unsafe_allow_html=True,
    )


def render_group_card(group_key: str):
    """Hiển thị thẻ thông tin 1 nhóm rác lớn (hữu cơ / vô cơ tái chế / vô cơ khác)."""
    group = GROUP_INFO[group_key]
    st.markdown(
        f"""
        <div style="border-top:4px solid {group['color']}; padding:14px 16px;
                    border-radius:6px; background:rgba(0,0,0,0.02); height:100%;">
            {icon_title(group['icon'], " <b>" + group['name'] + "</b>", size=26)}
            <p style="margin-top:10px; margin-bottom:6px;">{group['description']}</p>
            <p style="font-size:0.85em; opacity:0.8; margin:0;">🗑 {group['bin_note']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
def render_eco_note(class_key: str, eco_tip: str):
    """Thẻ thông điệp môi trường nhỏ, hiển thị ngay sau kết quả dự đoán.
    Dùng tông xanh lá/dương pastel cố định (không theo màu riêng của từng loại rác)
    để người dùng nhận diện nhất quán đây là 'thông điệp môi trường'."""
    st.markdown(
        f"""
        <div class="eco-note">
            <span style="opacity:0.9;">🌱</span>
            <span>{eco_tip}</span>
        </div>
        <style>
        .eco-note {{
            animation: ecoFadeIn 0.5s ease-out;
            background: rgba(46, 139, 87, 0.08);
            border-left: 3px solid #2E8B57;
            border-radius: 6px;
            padding: 10px 14px;
            margin: 12px 0;
            font-size: 0.9em;
            font-style: italic;
            opacity: 0.95;
            transition: background 0.25s ease;
        }}
        .eco-note:hover {{
            background: rgba(46, 139, 87, 0.13);
        }}
        @keyframes ecoFadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 0.95; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Câu quote môi trường cố định, hiển thị cuối mọi trang để tạo cảm giác thương hiệu nhất quán."""
    from src.predict import FOOTER_QUOTE

    st.markdown(
        f"""
        <div style="text-align:center; opacity:0.55; font-size:0.85em;
                    margin-top:36px; padding-top:14px; border-top:1px solid rgba(0,0,0,0.08);">
            {FOOTER_QUOTE}
        </div>
        """,
        unsafe_allow_html=True,
    )