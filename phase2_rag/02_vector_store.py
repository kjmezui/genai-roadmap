from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DOCS_PATH   = Path(__file__).parent / "documents"
CHROMA_PATH = Path(__file__).parent / "chroma_db"


# ── 1. CHARGEMENT ET CHUNKING ─────────────────────────────────
print("=" * 50)
print("1. CHARGEMENT ET CHUNKING")
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

print(f"Documents chargés : {len(documents)}")
print(f"Chunks produits   : {len(chunks)}")
print(f"Exemple métadonnées : {chunks[0].metadata}")


# ── 2. INDEXATION DANS CHROMADB ───────────────────────────────
print("\n" + "=" * 50)
print("2. INDEXATION DANS CHROMADB")
print("=" * 50)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Génération des embeddings et indexation...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_PATH),
    collection_name="regles_paiement",
)

print(f"✅ {vector_store._collection.count()} chunks indexés")
print(f"📁 Persisté dans : {CHROMA_PATH}")


# ── 3. RECHARGEMENT DEPUIS LE DISQUE ──────────────────────────
print("\n" + "=" * 50)
print("3. RECHARGEMENT DEPUIS LE DISQUE")
print("=" * 50)

vector_store_reloaded = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
    collection_name="regles_paiement",
)

print(f"✅ Collection rechargée : {vector_store_reloaded._collection.count()} chunks")


# ── 4. RECHERCHE SÉMANTIQUE ───────────────────────────────────
print("\n" + "=" * 50)
print("4. RECHERCHE SÉMANTIQUE")
print("=" * 50)

requetes = [
    "quel est le délai de remboursement en cas de fraude ?",
    "quels pays nécessitent une validation manuelle ?",
    "comment fonctionne l'authentification forte ?",
]

for requete in requetes:
    print(f"\n🔍 '{requete}'")
    resultats = vector_store_reloaded.similarity_search_with_score(
        query=requete,
        k=2,
    )
    for rang, (doc, score) in enumerate(resultats, 1):
        print(f"  #{rang} score={score:.4f} | chunk_id={doc.metadata['chunk_id']}")
        print(f"      {doc.page_content[:120]}...")


# ── 5. FILTRAGE PAR MÉTADONNÉE ────────────────────────────────
print("\n" + "=" * 50)
print("5. FILTRAGE PAR MÉTADONNÉE")
print("=" * 50)

resultats_filtres = vector_store_reloaded.similarity_search(
    query="règles de paiement",
    k=3,
    filter={"chunk_id": {"$lte": 2}},
)

print(f"Chunks récupérés avec filtre chunk_id ≤ 2 : {len(resultats_filtres)}")
for doc in resultats_filtres:
    print(f"  chunk_id={doc.metadata['chunk_id']} | {doc.page_content[:80]}...")