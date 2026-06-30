import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import List
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Détecte si on tourne dans Docker ou en local
DANS_DOCKER = os.path.exists("/app")

if DANS_DOCKER:
    CHROMA_PATH = Path("/app/phase2_rag/chroma_db")
    DOCS_PATH   = Path("/app/phase2_rag/documents")
else:
    CHROMA_PATH = Path(__file__).resolve().parents[1] / "phase2_rag" / "chroma_db"
    DOCS_PATH   = Path(__file__).resolve().parents[1] / "phase2_rag" / "documents"


# ── ÉTAT GLOBAL DE L'APPLICATION ──────────────────────────────
# En production, on évite les variables globales mutables, mais
# pour ce niveau pédagogique, c'est la façon la plus claire de
# comprendre le cycle de vie d'une API.

etat = {
    "chaine_rag":        None,
    "nb_requetes":       0,
    "latence_totale":    0.0,
    "demarrage":         None,
}


# ── CYCLE DE VIE DE L'APPLICATION ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code exécuté UNE SEULE FOIS au démarrage du serveur,
    puis une fois à l'arrêt. C'est ici qu'on charge le RAG
    en mémoire — pas à chaque requête.
    """
    print("🚀 Démarrage du service SENTINEL...")

    embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name="regles_paiement",
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2, "filter": {"chunk_id": {"$gt": 0}}},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    SYSTEM_PROMPT = """Tu es SENTINEL, un assistant expert en règles de paiement bancaire.
- Tu réponds UNIQUEMENT à partir du contexte fourni
- Si la réponse n'est pas dans le contexte, dis-le explicitement
- Tu structures ta réponse :
  RÉPONSE : [réponse directe]
  SOURCE   : [section du document]

CONTEXTE : {context}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def formater_contexte(docs: List[Document]) -> str:
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    etat["chaine_rag"] = (
        {
            "context":  RunnableLambda(lambda q: formater_contexte(
                            retriever.invoke(q))),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    etat["demarrage"] = time.time()

    print("✅ RAG chargé en mémoire — service prêt")

    yield  # le serveur tourne ici

    print("🛑 Arrêt du service SENTINEL")


app = FastAPI(
    title="SENTINEL RAG API",
    description="API de questions/réponses sur les règles de paiement bancaire",
    version="1.0.0",
    lifespan=lifespan,
)


# ── SCHÉMAS DE VALIDATION (Pydantic) ──────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Question sur les règles de paiement",
        examples=["Quel est le délai de remboursement en cas de fraude ?"],
    )


class ReponseRAG(BaseModel):
    reponse:        str
    latence_ms:     float
    nb_requetes:    int


class HealthStatus(BaseModel):
    status:         str
    rag_charge:     bool
    uptime_secondes: float


class Metrics(BaseModel):
    nb_requetes:        int
    latence_moyenne_ms: float
    uptime_secondes:    float


# ── ENDPOINTS ──────────────────────────────────────────────────

@app.post("/ask", response_model=ReponseRAG)
async def ask(payload: QuestionRequest):
    """Pose une question au système SENTINEL et reçoit une réponse sourcée."""

    if etat["chaine_rag"] is None:
        raise HTTPException(status_code=503, detail="RAG non initialisé")

    debut = time.time()
    try:
        reponse = etat["chaine_rag"].invoke(payload.question)
    except Exception as erreur:
        raise HTTPException(status_code=500, detail=f"Erreur génération : {erreur}")

    latence = (time.time() - debut) * 1000  # en millisecondes

    etat["nb_requetes"]    += 1
    etat["latence_totale"] += latence

    return ReponseRAG(
        reponse=reponse,
        latence_ms=round(latence, 2),
        nb_requetes=etat["nb_requetes"],
    )


@app.get("/health", response_model=HealthStatus)
async def health():
    """Vérifie que le service est opérationnel."""
    return HealthStatus(
        status="ok" if etat["chaine_rag"] else "degraded",
        rag_charge=etat["chaine_rag"] is not None,
        uptime_secondes=round(time.time() - etat["demarrage"], 1)
                         if etat["demarrage"] else 0.0,
    )


@app.get("/metrics", response_model=Metrics)
async def metrics():
    """Statistiques d'usage du service."""
    latence_moyenne = (
        etat["latence_totale"] / etat["nb_requetes"]
        if etat["nb_requetes"] > 0 else 0.0
    )
    return Metrics(
        nb_requetes=etat["nb_requetes"],
        latence_moyenne_ms=round(latence_moyenne, 2),
        uptime_secondes=round(time.time() - etat["demarrage"], 1)
                         if etat["demarrage"] else 0.0,
    )


@app.get("/")
async def racine():
    """Point d'entrée — redirige vers la documentation."""
    return {
        "service": "SENTINEL RAG API",
        "documentation": "/docs",
        "endpoints": ["/ask", "/health", "/metrics"],
    }