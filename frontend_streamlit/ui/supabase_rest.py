from datetime import datetime, timezone

import requests

from ui.app_config import get_supabase_settings
from ui.errors import ApiError


TABLE_URL_PATH = "/rest/v1/prediction_logs"
REQUEST_TIMEOUT = 15
HISTORY_COLUMNS = (
    "id,created_at,game_title,platform,genre,publisher,"
    "release_year,predicted_global_sales"
)


def _headers(settings):
    headers = {
        "apikey": settings.api_key,
        "Accept": "application/json",
    }
    if settings.use_bearer_token:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    return headers


def _response_error(status_code, operation):
    if status_code in (401, 403):
        permission = "SELECT" if operation == "read" else "INSERT"
        return ApiError(
            "Supabase menolak akses. Periksa anon/publishable key serta grant "
            f"dan kebijakan RLS untuk {permission} pada prediction_logs."
        )
    if status_code == 404:
        return ApiError(
            "Tabel prediction_logs tidak ditemukan pada Supabase Data API."
        )
    if status_code == 400:
        return ApiError(
            "Supabase menolak data. Periksa struktur prediction_logs dan "
            "kebijakan INSERT."
        )
    return ApiError("Supabase REST sedang tidak tersedia. Coba lagi nanti.")


def get_history():
    settings = get_supabase_settings()
    try:
        response = requests.get(
            f"{settings.url}{TABLE_URL_PATH}",
            params={
                "select": HISTORY_COLUMNS,
                "order": "created_at.desc,id.desc",
                "limit": "20",
            },
            headers=_headers(settings),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise ApiError(
            "Riwayat Supabase belum dapat dijangkau melalui HTTPS."
        ) from error
    if not response.ok:
        raise _response_error(response.status_code, "read")
    try:
        rows = response.json()
    except ValueError as error:
        raise ApiError("Supabase mengembalikan format riwayat yang tidak valid.") from error
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or any(column not in row for column in HISTORY_COLUMNS.split(","))
        for row in rows
    ):
        raise ApiError("Supabase mengembalikan struktur riwayat yang tidak valid.")
    return rows


def insert_prediction(prediction):
    settings = get_supabase_settings()
    created_at = datetime.now(timezone.utc).isoformat()
    record = {
        "created_at": created_at,
        "game_title": prediction["game_title"],
        "platform": prediction["platform"],
        "genre": prediction["genre"],
        "publisher": prediction["publisher"],
        "release_year": prediction["release_year"],
        "predicted_global_sales": prediction["predicted_global_sales"],
    }
    headers = _headers(settings)
    headers["Content-Type"] = "application/json"
    headers["Prefer"] = "return=minimal"
    try:
        response = requests.post(
            f"{settings.url}{TABLE_URL_PATH}",
            json=record,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise ApiError(
            "Prediksi berhasil dihitung, tetapi penyimpanan ke Supabase "
            "belum dapat dikonfirmasi."
        ) from error
    if not response.ok:
        raise _response_error(response.status_code, "write")
    return created_at
