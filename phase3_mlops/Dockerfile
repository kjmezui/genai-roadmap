# ── Image de base ──────────────────────────────────────────
FROM python:3.11-slim

# ── Répertoire de travail dans le container ──────────────────
WORKDIR /app

# ── Installation des dépendances d'abord (cache Docker) ──────
# Cette étape ne se reconstruit que si requirements.txt change
COPY phase3_mlops/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du code applicatif ──────────────────────────────────
COPY phase3_mlops/03_fastapi_app.py ./app.py
COPY phase2_rag/chroma_db ./phase2_rag/chroma_db
COPY phase2_rag/documents ./phase2_rag/documents

# ── Port exposé ────────────────────────────────────────────
EXPOSE 8000

# ── Vérification de santé automatique ─────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# ── Commande de démarrage ──────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]