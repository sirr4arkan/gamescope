import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")
DB_ENGINE = os.getenv("DB_ENGINE", "sqlite").strip().lower()
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "local-development-only-change-me")
DEBUG = os.getenv("DJANGO_DEBUG", "false").lower() == "true"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
    if host.strip()
]
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "predictor",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "config.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(BASE_DIR / "db.sqlite3")))

if DB_ENGINE == "postgresql":
    required_database_variables = ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST")
    missing_database_variables = [
        name for name in required_database_variables if not os.getenv(name, "").strip()
    ]
    if missing_database_variables:
        raise ImproperlyConfigured(
            "Missing required database environment variables: "
            + ", ".join(missing_database_variables)
        )
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("DB_NAME", ""),
            "USER": os.getenv("DB_USER", ""),
            "PASSWORD": os.getenv("DB_PASSWORD", ""),
            "HOST": os.getenv("DB_HOST", ""),
            "PORT": os.getenv("DB_PORT", "").strip() or "5432",
            "OPTIONS": {
                "sslmode": os.getenv("DB_SSLMODE", "").strip() or "require"
            },
            "CONN_MAX_AGE": 60,
        }
    }
elif DB_ENGINE == "sqlite":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": SQLITE_PATH,
        }
    }
else:
    raise ImproperlyConfigured(
        "Unsupported DB_ENGINE. Use 'sqlite' or 'postgresql'."
    )

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True
TIME_ZONE = "UTC"
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(ROOT_DIR / "artifacts" / "model_vgsales.joblib"))
)
CLEANED_DATA_PATH = Path(
    os.getenv(
        "CLEANED_DATA_PATH", str(ROOT_DIR / "artifacts" / "cleaned_vgsales.csv")
    )
)
MODEL_METADATA_PATH = Path(
    os.getenv(
        "MODEL_METADATA_PATH", str(ROOT_DIR / "artifacts" / "model_metadata.json")
    )
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"
