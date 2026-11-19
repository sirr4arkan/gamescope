import json

from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    Groq,
    RateLimitError,
)

from ui.app_config import get_groq_settings
from ui.errors import ApiError
from ui.local_inference import get_metadata


SYSTEM_PROMPT = (
    "Kamu adalah tutor analisis data yang membantu mahasiswa D3 memahami grafik "
    "historis GameScope. Jawab hanya dalam bahasa Indonesia yang natural, jelas, "
    "ringkas, dan menggunakan Markdown. Analisis hanya angka dalam konteks JSON "
    "yang diberikan: sebutkan kategori tertinggi dan terendah, bandingkan nilai "
    "secara konkret, lalu jelaskan pola distribusi dan batas interpretasinya. "
    "Jangan mengarang angka, jangan menyatakan hubungan sebab-akibat, jangan "
    "menilai kualitas game dari penjualannya, dan jangan membuat prediksi masa "
    "depan. Tegaskan bahwa ini agregat historis; rata-rata dapat dipengaruhi "
    "jumlah observasi dan beberapa judul dengan penjualan tinggi. Batasi jawaban "
    "sekitar 180 kata."
)

ANALYSIS_INSTRUCTIONS = {
    "genre_analysis": (
        "Analisis grafik rata-rata Global Sales berdasarkan genre. Bandingkan "
        "genre tertinggi dan terendah beserta nilai dan jumlah observasinya, "
        "jelaskan pola antarkategori secara sederhana, dan ingatkan bahwa hasil "
        "merupakan rata-rata deskriptif."
    ),
    "platform_analysis": (
        "Analisis grafik rata-rata Global Sales berdasarkan platform. Bandingkan "
        "platform tertinggi dan terendah beserta nilai dan jumlah observasinya, "
        "jelaskan pola antarkategori secara sederhana, dan ingatkan bahwa hasil "
        "merupakan rata-rata deskriptif."
    ),
}
MAX_COMPLETION_TOKENS = 450
MAX_EXPLANATION_CHARACTERS = 8000


def explain_market_analysis(analysis_type):
    instruction = ANALYSIS_INSTRUCTIONS.get(analysis_type)
    if instruction is None:
        raise ApiError("Jenis analisis grafik tidak dikenali.")
    metadata = get_metadata()
    context = metadata["sales_analysis"][analysis_type]
    settings = get_groq_settings()
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))

    try:
        with Groq(
            api_key=settings.api_key,
            timeout=30.0,
            max_retries=0,
        ) as client:
            completion = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{instruction}\n\n"
                            "Gunakan kategori, nilai rata-rata, dan jumlah observasi "
                            "persis dari data berikut. Angka penjualan memakai satuan "
                            "juta unit terjual; rentang tahun dan total observasi juga "
                            "disertakan. Jangan memperlakukan data sebagai bukti "
                            "sebab-akibat.\n"
                            f"Data aktual:\n{context_json}"
                        ),
                    },
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
    except APITimeoutError as error:
        raise ApiError(
            "Penjelasan grafik memerlukan waktu lebih lama. Silakan coba lagi."
        ) from error
    except RateLimitError as error:
        raise ApiError(
            "Layanan AI sedang sibuk. Silakan coba lagi sebentar lagi."
        ) from error
    except APIStatusError as error:
        if error.status_code in (401, 403):
            raise ApiError(
                "Groq menolak kredensial. Periksa GROQ_API_KEY di Streamlit Secrets."
            ) from error
        raise ApiError(
            "Penjelasan grafik belum dapat dibuat. Silakan coba lagi nanti."
        ) from error
    except (APIConnectionError, APIError) as error:
        raise ApiError(
            "Layanan penjelasan grafik belum dapat dijangkau melalui HTTPS."
        ) from error

    try:
        explanation = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise ApiError(
            "Jawaban AI belum dapat ditampilkan. Silakan coba lagi."
        ) from error
    if (
        not isinstance(explanation, str)
        or not explanation.strip()
        or len(explanation) > MAX_EXPLANATION_CHARACTERS
    ):
        raise ApiError("Jawaban AI belum dapat ditampilkan. Silakan coba lagi.")
    return explanation.strip()
