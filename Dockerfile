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

# Use shell form so env vars expand on Render and elsewhere.
CMD sh -c 'python -m gto_poker_trainer_cli.ui.textual_serve_main --host 0.0.0.0 --port ${PORT:-8000} --hands ${HANDS:-1} --mc ${MC:-200}'
