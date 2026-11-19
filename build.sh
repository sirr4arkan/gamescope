#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python backend_django/manage.py collectstatic --noinput
python backend_django/manage.py migrate --noinput
