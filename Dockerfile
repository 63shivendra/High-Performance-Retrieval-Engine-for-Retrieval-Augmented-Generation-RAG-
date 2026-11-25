# Live demo image for SelfIndex.
# Works on Render, Hugging Face Spaces (Docker), Railway, Fly.io, or locally:
#   docker build -t selfindex . && docker run -p 7860:7860 selfindex
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/share/nltk_data \
    PORT=7860

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt \
 && python -m nltk.downloader -d $NLTK_DATA stopwords punkt punkt_tab

COPY src ./src
COPY app ./app
COPY demo_data/corpus.jsonl.gz ./demo_data/
COPY plots ./plots
COPY results ./results

# Tokenize the demo corpus at build time so the container starts in seconds.
RUN python -m app.engine prepare

# Run as a non-root user (required by Hugging Face Spaces, good practice elsewhere).
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/healthz')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
