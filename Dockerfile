FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . /app

RUN python -m pip install -U pip \
 && python -m pip install -e '.[tui]'

EXPOSE 8000

# Use shell form so env vars expand on Render and elsewhere.
CMD sh -c 'gto-poker-trainer-serve --host 0.0.0.0 --port ${PORT:-8000} --hands ${HANDS:-1} --mc ${MC:-200}'
