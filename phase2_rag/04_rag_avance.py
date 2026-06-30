from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from typing import List
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CHROMA_PATH = Path(__file__).parent / "chroma_db"
DOCS_PATH   = Path(__file__).parent / "documents"


# ── 1. CHARGEMENT DES CHUNKS ──────────────────────────────────
print("=" * 50)
print("1. CHARGEMENT DES CHUNKS")
print("=" * 50)

loader = TextLoader(str(DOCS_PATH / "regles_paiement.txt"), encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["source"]   = "regles_paiement"

chunks_utiles = [c for c in chunks if c.metadata["chunk_id"] > 0]
print(f"Chunks totaux  : {len(chunks)}")
print(f"Chunks utiles  : {len(chunks_utiles)} (titre exclu)")


# ── 2. RETRIEVERS ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. RETRIEVERS")
print("=" * 50)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
    collection_name="regles_paiement",
)

retriever_dense = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
        "filter": {"chunk_id": {"$gt": 0}},
    },
)

retriever_bm25 = BM25Retriever.from_documents(chunks_utiles)
retriever_bm25.k = 4

print("✅ Retriever dense (ChromaDB) configuré")
print("✅ Retriever sparse (BM25) configuré")


# ── 3. FUSION MANUELLE (remplace EnsembleRetriever) ───────────
print("\n" + "=" * 50)
print("3. RETRIEVER HYBRIDE MANUEL")
print("=" * 50)

class HybridRetriever(BaseRetriever):
    """Fusionne dense + BM25 avec pondération puis reranke via Cohere."""

    retriever_dense: object = Field(...)
    retriever_bm25:  object = Field(...)
    reranker:        object = Field(...)
    poids_dense:     float  = Field(default=0.6)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Récupération depuis les deux retrievers
        docs_dense = self.retriever_dense.invoke(query)
        docs_bm25  = self.retriever_bm25.invoke(query)

        # Fusion avec déduplication par contenu
        vus   = set()
        fusionnes = []

        for doc in docs_dense:
            cle = doc.page_content[:80]
            if cle not in vus:
                vus.add(cle)
                fusionnes.append(doc)

        for doc in docs_bm25:
            cle = doc.page_content[:80]
            if cle not in vus:
                vus.add(cle)
                fusionnes.append(doc)

        # Reranking Cohere sur la liste fusionnée
        rerankes = self.reranker.compress_documents(fusionnes, query)
        return rerankes

reranker = CohereRerank(
    model="rerank-multilingual-v3.0",
    top_n=3,
)

retriever_final = HybridRetriever(
    retriever_dense=retriever_dense,
    retriever_bm25=retriever_bm25,
    reranker=reranker,
)

print("✅ Retriever hybride manuel configuré (dense + BM25 + Cohere rerank)")


# ── 4. TEST DU PIPELINE DE RETRIEVAL ─────────────────────────
print("\n" + "=" * 50)
print("4. TEST COMPARATIF")
print("=" * 50)

questions_test = [
    "délai remboursement fraude avérée",
    "authentification 3DS2 transaction en ligne",
    "pays GAFI validation manuelle",
]

for question in questions_test:
    print(f"\n🔍 '{question}'")
    docs_dense  = retriever_dense.invoke(question)
    docs_finaux = retriever_final.invoke(question)
    print(f"  Dense seul  → chunks : {[d.metadata['chunk_id'] for d in docs_dense]}")
    print(f"  Hybride+RR  → chunks : {[d.metadata['chunk_id'] for d in docs_finaux]}")


# ── 5. CHAÎNE RAG AVANCÉE ─────────────────────────────────────
print("\n" + "=" * 50)
print("5. CHAÎNE RAG AVANCÉE")
print("=" * 50)

SYSTEM_PROMPT = """Tu es SENTINEL, un assistant expert en règles de paiement bancaire.

RÈGLES STRICTES :
- Tu réponds UNIQUEMENT à partir du contexte fourni
- Si la réponse n'est pas dans le contexte, réponds exactement :
  "Je ne trouve pas cette information dans les règles de paiement disponibles."
- Tu structures ta réponse ainsi :
  RÉPONSE : [réponse directe et précise]
  SOURCE   : [section du document source]
- Ton ton est professionnel et factuel

CONTEXTE :
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def formater_contexte(docs: List[Document]) -> str:
    sections = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "?")
        lignes   = doc.page_content.strip().split("\n")
        titre    = lignes[0] if lignes[0].startswith("Section") else f"chunk #{chunk_id}"
        sections.append(f"[{titre}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)

chaine_rag_avancee = (
    {
        "context":  RunnableLambda(lambda q: formater_contexte(
                        retriever_final.invoke(q))),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Chaîne RAG avancée construite\n")

questions = [
    "Quel est le délai de remboursement en cas de fraude avérée ?",
    "Quels sont les pays qui nécessitent une validation manuelle ?",
    "Quelles sont les règles pour les virements SEPA et hors SEPA ?",
    "Quelle est la politique de remboursement pour les retards de livraison ?",
]

for question in questions:
    print(f"\n{'─' * 50}")
    print(f"❓ {question}")
    print(f"{'─' * 50}")
    reponse = chaine_rag_avancee.invoke(question)
    print(f"🤖 {reponse}")