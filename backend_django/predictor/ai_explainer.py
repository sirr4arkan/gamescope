import json

from django.conf import settings
from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    Groq,
    RateLimitError,
)


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


class AIExplanationError(Exception):
    status_code = 502
    error_code = "provider_unavailable"
    public_message = "Penjelasan AI belum dapat dibuat. Silakan coba lagi nanti."


class UnknownAnalysisType(AIExplanationError):
    status_code = 400
    error_code = "unknown_analysis"
    public_message = "Jenis analisis tidak dikenali."


class MissingGroqConfiguration(AIExplanationError):
    status_code = 503
    error_code = "missing_configuration"
    public_message = "Fitur penjelasan AI belum dikonfigurasi pada backend."


class GroqRateLimited(AIExplanationError):
    status_code = 429
    error_code = "rate_limited"
    public_message = "Layanan AI sedang sibuk. Silakan coba lagi sebentar lagi."


class GroqRequestTimedOut(AIExplanationError):
    status_code = 504
    error_code = "timeout"
    public_message = "Penjelasan AI memerlukan waktu lebih lama. Silakan coba lagi."


class InvalidAIResponse(AIExplanationError):
    error_code = "invalid_response"
    public_message = "Layanan AI memberikan jawaban yang tidak dapat digunakan. Coba lagi."


def generate_analysis_explanation(analysis_type, analysis_context):
    instruction = ANALYSIS_INSTRUCTIONS.get(analysis_type)
    if instruction is None:
        raise UnknownAnalysisType
    if not settings.GROQ_API_KEY:
        raise MissingGroqConfiguration

    context_json = json.dumps(analysis_context, ensure_ascii=False, separators=(",", ":"))
    try:
        with Groq(
            api_key=settings.GROQ_API_KEY,
            timeout=30.0,
            max_retries=0,
        ) as client:
            completion = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{instruction}\n\n"
                            "Gunakan kategori, nilai rata-rata, dan jumlah observasi "
                            "persis dari data berikut. Angka penjualan memakai satuan "
                            "juta unit terjual; rentang tahun dan total observasi juga disertakan. "
                            "Jangan memperlakukan data sebagai bukti sebab-akibat.\n"
                            f"Data aktual:\n{context_json}"
                        ),
                    },
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
    except APITimeoutError as error:
        raise GroqRequestTimedOut from error
    except RateLimitError as error:
        raise GroqRateLimited from error
    except APIConnectionError as error:
        raise AIExplanationError from error
    except APIStatusError as error:
        if error.status_code in (401, 403):
            raise MissingGroqConfiguration from error
        raise AIExplanationError from error
    except APIError as error:
        raise AIExplanationError from error

    try:
        explanation = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise InvalidAIResponse from error
    if not isinstance(explanation, str) or not explanation.strip():
        raise InvalidAIResponse
    explanation = explanation.strip()
    if len(explanation) > MAX_EXPLANATION_CHARACTERS:
        raise InvalidAIResponse
    return explanation
