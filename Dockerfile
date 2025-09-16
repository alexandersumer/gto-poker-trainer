FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . /app

RUN python -m pip install -U pip \
 && python -m pip install -e . \
 && python -m pip install 'textual>=0.60' 'textual-serve>=0.3'

EXPOSE 8000

# Run via Textual's built-in web server to load the app directly (no landing page).
CMD sh -c 'python -m textual serve "python -m gto_poker_trainer_cli.ui.textual_main --hands ${HANDS:-1} --mc ${MC:-200}" --host 0.0.0.0 --port ${PORT:-8000}'
