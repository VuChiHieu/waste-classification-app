"""
Tiện ích hiển thị icon Lucide (https://lucide.dev) trong Streamlit.

Cách hoạt động: Lucide phát hành package `lucide-static` chứa từng icon dưới
dạng file .svg riêng lẻ, phân phối qua CDN unpkg. Ta chỉ cần nhúng thẻ <img>
trỏ tới URL đó — trình duyệt người dùng sẽ tự tải icon, không cần cài đặt gì
trong môi trường Python/Streamlit Cloud.

Icon set (ISC License, mã nguồn mở, dùng tự do): https://lucide.dev
"""

LUCIDE_CDN = "https://unpkg.com/lucide-static@latest/icons/{name}.svg"


def icon_html(name: str, size: int = 20, opacity: float = 1.0) -> str:
    """Trả về chuỗi HTML <img> cho 1 icon Lucide, dùng trong st.markdown(unsafe_allow_html=True)."""
    style = f"vertical-align:-4px; margin-right:6px; opacity:{opacity};"
    return f'<img src="{LUCIDE_CDN.format(name=name)}" width="{size}" height="{size}" style="{style}" />'


def icon_title(name: str, text: str, size: int = 22, tag: str = "span") -> str:
    """Icon + chữ trên cùng 1 dòng, dùng cho tiêu đề thẻ thông tin."""
    return f'<{tag}>{icon_html(name, size=size)}{text}</{tag}>'