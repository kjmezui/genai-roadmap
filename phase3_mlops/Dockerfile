FROM python:3.11-slim

WORKDIR /app

COPY phase3_mlops/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY phase3_mlops/03_fastapi_app.py ./app.py
COPY phase2_rag/documents ./phase2_rag/documents

# chroma_db est monté via volume au runtime — pas copié dans l'image
# En CI, on vérifie juste que l'image se construit correctement

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]