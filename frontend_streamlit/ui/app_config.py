import base64
import binascii
import json
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ui.errors import ApiError


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    api_key: str = field(repr=False)
    use_bearer_token: bool


@dataclass(frozen=True)
class GroqSettings:
    api_key: str = field(repr=False)
    model: str


def _get_secret(name):
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return ""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ApiError(f"Konfigurasi {name} harus berupa teks.")
    return value.strip()


def _is_service_role_key(api_key):
    if api_key.startswith("sb_secret_"):
        return True

    parts = api_key.split(".")
    if len(parts) != 3:
        return False
    payload = parts[1]
    try:
        decoded = base64.urlsafe_b64decode(
            payload + "=" * (-len(payload) % 4)
        )
        claims = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return False
    return isinstance(claims, dict) and claims.get("role") == "service_role"


def get_supabase_settings():
    url = _get_secret("SUPABASE_URL").rstrip("/")
    api_key = _get_secret("SUPABASE_ANON_KEY") or _get_secret(
        "SUPABASE_PUBLISHABLE_KEY"
    )
    if not url:
        raise ApiError(
            "Supabase belum dikonfigurasi. Tambahkan SUPABASE_URL ke Streamlit Secrets."
        )
    if not api_key:
        raise ApiError(
            "Supabase belum dikonfigurasi. Tambahkan SUPABASE_ANON_KEY "
            "(atau SUPABASE_PUBLISHABLE_KEY) ke Streamlit Secrets."
        )

    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname or ""
        port = parsed_url.port
    except ValueError as error:
        raise ApiError(
            "SUPABASE_URL harus berupa URL HTTPS project Supabase yang valid."
        ) from error
    if (
        parsed_url.scheme != "https"
        or parsed_url.username
        or parsed_url.password
        or port not in (None, 443)
        or parsed_url.path not in ("", "/")
        or parsed_url.query
        or parsed_url.fragment
        or not hostname.endswith((".supabase.co", ".supabase.com"))
    ):
        raise ApiError(
            "SUPABASE_URL harus berupa URL HTTPS project Supabase yang valid."
        )
    if _is_service_role_key(api_key):
        raise ApiError(
            "Gunakan Supabase anon/publishable key; service-role key tidak boleh "
            "dipakai oleh aplikasi Streamlit."
        )

    return SupabaseSettings(
        url=url,
        api_key=api_key,
        use_bearer_token=not api_key.startswith("sb_publishable_"),
    )


def get_groq_settings():
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        raise ApiError(
            "Fitur penjelasan AI belum dikonfigurasi. Tambahkan GROQ_API_KEY "
            "ke Streamlit Secrets."
        )
    model = _get_secret("GROQ_MODEL") or "openai/gpt-oss-20b"
    return GroqSettings(api_key=api_key, model=model)
