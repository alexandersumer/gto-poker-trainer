FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . /app

RUN python -m pip install -U pip \
 && python -m pip install -e . \
 && python -m pip install fastapi uvicorn

EXPOSE 8000

# Run FastAPI web app (no Textual dependency for serving).
CMD ["python","-m","gto_poker_trainer.web.app"]
