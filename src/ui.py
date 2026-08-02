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