FROM python:3.13.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . /app

RUN python -m pip install -U pip \
 && python -m pip install --no-cache-dir "Cython>=3.0" "setuptools>=70" wheel \
 && python -m pip install --no-cache-dir -e . \
 && python -m pip install --no-cache-dir fastapi uvicorn

EXPOSE 8000

# Run FastAPI web app (no Textual dependency for serving).
CMD ["python","-m","gtotrainer.web.app"]
