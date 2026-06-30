from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CHROMA_PATH = Path(__file__).parent / "chroma_db"


# ── 1. CHARGEMENT DU VECTOR STORE ─────────────────────────────
print("=" * 50)
print("1. CHARGEMENT DU VECTOR STORE")
print("=" * 50)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
    collection_name="regles_paiement",
)

print(f"✅ {vector_store._collection.count()} chunks chargés depuis le disque")


# ── 2. CONFIGURATION DU RETRIEVER ─────────────────────────────
print("\n" + "=" * 50)
print("2. CONFIGURATION DU RETRIEVER")
print("=" * 50)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # on récupère les 3 chunks les plus proches
)

# Test du retriever seul
docs = retriever.invoke("délai de remboursement fraude")
print(f"Chunks retrievés : {len(docs)}")
for i, doc in enumerate(docs):
    print(f"  #{i+1} chunk_id={doc.metadata['chunk_id']} | "
          f"{doc.page_content[:80]}...")


# ── 3. PROMPT TEMPLATE ────────────────────────────────────────
print("\n" + "=" * 50)
print("3. PROMPT TEMPLATE")
print("=" * 50)

SYSTEM_PROMPT = """Tu es SENTINEL, un assistant expert en règles de paiement bancaire.

RÈGLES STRICTES :
- Tu réponds UNIQUEMENT à partir du contexte fourni ci-dessous
- Si la réponse n'est pas dans le contexte, tu dis explicitement :
  "Je ne trouve pas cette information dans les règles de paiement disponibles."
- Tu cites toujours la section source de ta réponse
- Ton ton est professionnel et précis
- Tu ne formules jamais d'avis personnel

CONTEXTE :
{context}
"""

HUMAN_PROMPT = "Question : {question}"

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])

print("✅ Prompt template configuré")
print(f"   Variables attendues : {prompt.input_variables}")


# ── 4. CONSTRUCTION DE LA CHAÎNE RAG ──────────────────────────
print("\n" + "=" * 50)
print("4. CONSTRUCTION DE LA CHAÎNE RAG")
print("=" * 50)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,  # très bas : réponses factuelles et stables
)

def formater_contexte(docs) -> str:
    """Formate les chunks retrievés en un contexte lisible pour le LLM."""
    return "\n\n---\n\n".join([
        f"[Section source : chunk #{doc.metadata['chunk_id']}]\n{doc.page_content}"
        for doc in docs
    ])

# Construction de la chaîne LCEL (LangChain Expression Language)
chaine_rag = (
    {
        "context":  retriever | formater_contexte,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Chaîne RAG construite")


# ── 5. QUESTIONS / RÉPONSES ───────────────────────────────────
print("\n" + "=" * 50)
print("5. QUESTIONS / RÉPONSES")
print("=" * 50)

questions = [
    "Quel est le délai de remboursement en cas de fraude avérée ?",
    "Quels sont les plafonds de retrait pour une carte premium ?",
    "Comment fonctionne l'authentification forte pour les achats en ligne ?",
    "Quel est le taux de change appliqué pour les paiements en devise étrangère ?",
    "Quelle est la politique de remboursement pour les retards de livraison ?",  # hors contexte
]

for question in questions:
    print(f"\n{'─' * 50}")
    print(f"❓ {question}")
    print(f"{'─' * 50}")
    reponse = chaine_rag.invoke(question)
    print(f"🤖 {reponse}")


# ── 6. MODE STREAMING ─────────────────────────────────────────
print("\n" + "=" * 50)
print("6. MODE STREAMING")
print("=" * 50)

question_stream = "Quelles sont les règles pour les virements internationaux ?"
print(f"❓ {question_stream}\n")
print("🤖 ", end="", flush=True)

for chunk in chaine_rag.stream(question_stream):
    print(chunk, end="", flush=True)

print("\n")