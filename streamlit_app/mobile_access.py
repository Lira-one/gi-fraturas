"""Acesso pelo celular: descobre o endereço do app na rede local e gera um QR
code apontando para ele.

O celular precisa estar na mesma rede Wi-Fi do computador que roda o
Streamlit (o servidor escuta em 0.0.0.0, ver .streamlit/config.toml). Para
apresentar fora da rede local (app publicado no Streamlit Community Cloud,
túnel ngrok etc.), defina a variável de ambiente GI_PUBLIC_URL com o
endereço público e o QR code passa a apontar para ele.
"""
import base64
import io
import os
import socket
from urllib.parse import urlparse

import qrcode
import streamlit as st
from qrcode.constants import ERROR_CORRECT_M

from theme import PRIMARY_DARK, render_html


def _lan_ip() -> str:
    """IP desta máquina na rede local. O connect() em UDP não envia nenhum
    pacote: só faz o sistema escolher a interface de saída."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def app_url() -> str:
    public_url = os.environ.get("GI_PUBLIC_URL", "").strip()
    if public_url:
        return public_url.rstrip("/")
    # Se o app já foi aberto por um endereço público (ex.: Streamlit Cloud),
    # usa esse mesmo endereço no QR code.
    try:
        current = urlparse(st.context.url or "")
    except Exception:
        current = urlparse("")
    if current.hostname and current.hostname not in ("localhost", "127.0.0.1", "0.0.0.0"):
        return f"{current.scheme}://{current.netloc}"
    port = st.get_option("server.port") or 8501
    return f"http://{_lan_ip()}:{port}"


@st.cache_data(show_spinner=False)
def qr_code_png(url: str) -> bytes:
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color=PRIMARY_DARK, back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def render_mobile_access(variant: str = "card"):
    """variant="card": cartão horizontal (tela de login, oculto no celular —
    quem já está no celular não precisa do QR). variant="sidebar": versão
    vertical para a barra lateral."""
    url = app_url()
    b64 = base64.b64encode(qr_code_png(url)).decode("ascii")
    extra_class = "gi-qr-sidebar" if variant == "sidebar" else "gi-hide-mobile"
    is_local_only = url.startswith("http://127.") or "localhost" in url
    is_public = url.startswith("https://")
    if is_local_only:
        hint = "Não foi possível detectar a rede local; conecte o computador a uma rede Wi-Fi."
    elif is_public:
        hint = "Aponte a câmera do celular para o código para abrir o app."
    else:
        hint = "Aponte a câmera do celular para o código. O aparelho precisa estar na mesma rede Wi-Fi."
    render_html(
        f"""
        <div class="gi-qr {extra_class}">
            <img src="data:image/png;base64,{b64}" alt="QR code para abrir o app no celular"/>
            <div>
                <div class="gi-qr-title">📱 Acessar pelo celular</div>
                <div class="gi-qr-text">{hint}</div>
                <span class="gi-qr-url">{url}</span>
            </div>
        </div>
        """
    )
