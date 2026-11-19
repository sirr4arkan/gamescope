import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("DJANGO_API_URL", "http://127.0.0.1:8000/api").rstrip("/")
REQUEST_TIMEOUT = 10


class ApiError(Exception):
    """A user-safe error returned by an internal application service."""


def _request_error(response, fallback):
    if response.status_code == 400:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            return ApiError(body["error"])
        return ApiError("Periksa kembali data yang dimasukkan.")
    return ApiError(fallback)


@st.cache_data(ttl=300)
def get_metadata():
    try:
        response = requests.get(
            f"{API_BASE_URL}/metadata/", timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as error:
        raise ApiError(
            "Data aplikasi belum dapat dimuat. Pastikan layanan tersedia, lalu coba lagi."
        ) from error
    if not response.ok:
        raise _request_error(
            response, "Data aplikasi belum dapat dimuat. Coba lagi nanti."
        )
    try:
        metadata = response.json()
    except ValueError as error:
        raise ApiError("Data aplikasi belum dapat dimuat. Coba lagi nanti.") from error

    required_lists = ("platforms", "genres", "publishers")
    if not isinstance(metadata, dict):
        raise ApiError("Data aplikasi belum tersedia. Coba lagi nanti.")
    for key in required_lists:
        values = metadata.get(key)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) for value in values)
        ):
            raise ApiError("Data aplikasi belum tersedia. Coba lagi nanti.")
    year_range = metadata.get("year_range")
    if (
        not isinstance(year_range, list)
        or len(year_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in year_range
        )
        or year_range[0] > year_range[1]
    ):
        raise ApiError("Data aplikasi belum tersedia. Coba lagi nanti.")
    for key in ("average_sales_by_genre", "average_sales_by_platform"):
        if not isinstance(metadata.get(key), dict):
            raise ApiError("Data aplikasi belum tersedia. Coba lagi nanti.")
    if not isinstance(metadata.get("dataset_summary"), dict) or not isinstance(
        metadata.get("model_summary"), dict
    ):
        raise ApiError("Data aplikasi belum tersedia. Coba lagi nanti.")
    return metadata


def get_history():
    try:
        response = requests.get(
            f"{API_BASE_URL}/history/", timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as error:
        raise ApiError(
            "Riwayat belum dapat dimuat. Pastikan layanan tersedia, lalu coba lagi."
        ) from error
    if not response.ok:
        raise _request_error(response, "Riwayat belum dapat dimuat. Coba lagi nanti.")
    try:
        body = response.json()
    except ValueError as error:
        raise ApiError("Riwayat belum dapat dimuat. Coba lagi nanti.") from error
    if not isinstance(body, dict) or not isinstance(body.get("history"), list):
        raise ApiError("Riwayat belum dapat dimuat. Coba lagi nanti.")
    return body["history"]


def predict(payload):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise ApiError(
            "Prediksi belum dapat diproses. Pastikan layanan tersedia, lalu coba lagi."
        ) from error
    if not response.ok:
        raise _request_error(response, "Prediksi belum dapat diproses. Coba lagi nanti.")
    try:
        result = response.json()
    except ValueError as error:
        raise ApiError("Hasil prediksi belum dapat ditampilkan. Coba lagi nanti.") from error
    if not isinstance(result, dict):
        raise ApiError("Hasil prediksi belum dapat ditampilkan. Coba lagi nanti.")
    sales = result.get("predicted_global_sales")
    category = result.get("category")
    if (
        isinstance(sales, bool)
        or not isinstance(sales, (int, float))
        or not isinstance(category, str)
        or category not in ("Niche", "Standard", "Blockbuster")
    ):
        raise ApiError("Hasil prediksi belum dapat ditampilkan. Coba lagi nanti.")
    return result


def explain_market_analysis(analysis_type):
    if analysis_type not in ("genre_analysis", "platform_analysis"):
        raise ApiError("Jenis analisis grafik tidak dikenali.")

    try:
        response = requests.post(
            f"{API_BASE_URL}/explain/",
            json={"diagram_type": analysis_type},
            timeout=35,
        )
    except requests.Timeout as error:
        raise ApiError(
            "Penjelasan grafik memerlukan waktu lebih lama. Silakan coba lagi."
        ) from error
    except requests.RequestException as error:
        raise ApiError(
            "Layanan penjelasan grafik belum dapat dijangkau. Coba lagi nanti."
        ) from error

    if not response.ok:
        try:
            body = response.json()
        except ValueError:
            body = None
        if (
            response.status_code in (400, 429, 502, 503, 504)
            and isinstance(body, dict)
            and isinstance(body.get("error"), str)
        ):
            raise ApiError(body["error"])
        raise ApiError(
            "Penjelasan grafik belum dapat dibuat. Silakan coba lagi nanti."
        )

    try:
        result = response.json()
    except ValueError as error:
        raise ApiError(
            "Jawaban AI belum dapat ditampilkan. Silakan coba lagi."
        ) from error
    if (
        not isinstance(result, dict)
        or result.get("diagram_type") != analysis_type
        or not isinstance(result.get("explanation"), str)
        or not result["explanation"].strip()
        or len(result["explanation"]) > 8000
    ):
        raise ApiError("Jawaban AI belum dapat ditampilkan. Silakan coba lagi.")
    return result["explanation"].strip()
