import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")


def _environment_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ImproperlyConfigured(f"{name} must be a boolean value.")


IS_RENDER = _environment_bool("RENDER", False) or bool(
    os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
)
DB_ENGINE = os.getenv("DB_ENGINE", "sqlite").strip().lower()
DEBUG = _environment_bool("DJANGO_DEBUG", False)
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "").strip()
if SECRET_KEY in {
    "",
    "local-development-only-change-me",
    "replace-with-a-local-secret",
    "replace-with-a-random-secret",
}:
    if DEBUG and not IS_RENDER:
        SECRET_KEY = "local-development-only-change-me"
    else:
        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY must be set to a non-placeholder value."
        )
if IS_RENDER and DEBUG:
    raise ImproperlyConfigured("DJANGO_DEBUG must be false on Render.")

ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
    if host.strip()
]
RENDER_EXTERNAL_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
if RENDER_EXTERNAL_HOSTNAME and RENDER_EXTERNAL_HOSTNAME not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)
if IS_RENDER and "*" in ALLOWED_HOSTS:
    raise ImproperlyConfigured("Wildcard ALLOWED_HOSTS is not allowed on Render.")

CSRF_TRUSTED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]
if RENDER_EXTERNAL_HOSTNAME:
    render_origin = f"https://{RENDER_EXTERNAL_HOSTNAME}"
    if render_origin not in CSRF_TRUSTED_ORIGINS:
        CSRF_TRUSTED_ORIGINS.append(render_origin)

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
    "whitenoise.middleware.WhiteNoiseMiddleware",
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
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

SECURE_PROXY_SSL_HEADER = (
    ("HTTP_X_FORWARDED_PROTO", "https") if IS_RENDER else None
)
SECURE_SSL_REDIRECT = _environment_bool("DJANGO_SECURE_SSL_REDIRECT", IS_RENDER)
SESSION_COOKIE_SECURE = IS_RENDER
CSRF_COOKIE_SECURE = IS_RENDER
try:
    SECURE_HSTS_SECONDS = int(
        os.getenv("DJANGO_SECURE_HSTS_SECONDS", "31536000" if IS_RENDER else "0")
    )
except ValueError as error:
    raise ImproperlyConfigured(
        "DJANGO_SECURE_HSTS_SECONDS must be an integer."
    ) from error
if SECURE_HSTS_SECONDS < 0:
    raise ImproperlyConfigured("DJANGO_SECURE_HSTS_SECONDS cannot be negative.")
SECURE_HSTS_INCLUDE_SUBDOMAINS = _environment_bool(
    "DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", False
)
SECURE_HSTS_PRELOAD = _environment_bool("DJANGO_SECURE_HSTS_PRELOAD", False)
SECURE_CONTENT_TYPE_NOSNIFF = True

SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(BASE_DIR / "db.sqlite3")))

if IS_RENDER and DB_ENGINE != "postgresql":
    raise ImproperlyConfigured("Render must use PostgreSQL via DB_ENGINE=postgresql.")

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
