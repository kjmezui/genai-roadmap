import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from pydantic import Field
from typing import List
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CHROMA_PATH = Path(__file__).resolve().parents[1] / "phase2_rag" / "chroma_db"
DOCS_PATH   = Path(__file__).resolve().parents[1] / "phase2_rag" / "documents"


# ── 1. VÉRIFICATION LANGSMITH ─────────────────────────────────
print("=" * 50)
print("1. VÉRIFICATION LANGSMITH")
print("=" * 50)

api_key  = os.getenv("LANGCHAIN_API_KEY", "")
tracing  = os.getenv("LANGCHAIN_TRACING_V2", "")
project  = os.getenv("LANGCHAIN_PROJECT", "")

print(f"API Key   : {'✅ configurée (' + api_key[:10] + '...)' if api_key else '❌ manquante'}")
print(f"Tracing   : {'✅ activé' if tracing == 'true' else '❌ désactivé'}")
print(f"Project   : {project if project else '❌ non défini'}")

if not api_key:
    print("\n⚠️  Ajoutez LANGCHAIN_API_KEY dans votre .env et relancez.")
    exit(1)


# ── 2. RECONSTRUCTION DU PIPELINE OPTIMISÉ ───────────────────
print("\n" + "=" * 50)
print("2. PIPELINE RAG OPTIMISÉ (chunk=400, k=2)")
print("=" * 50)

loader = TextLoader(str(DOCS_PATH / "regles_paiement.txt"), encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,    # meilleure config identifiée par MLflow
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["source"]   = "regles_paiement"

embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
    collection_name="regles_paiement",
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2,  # k=2 optimal selon MLflow
        "filter": {"chunk_id": {"$gt": 0}},
    },
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    tags=["sentinel-llm"],          # tag visible dans LangSmith
)

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
    return "\n\n---\n\n".join([
        f"[chunk #{doc.metadata['chunk_id']}]\n{doc.page_content}"
        for doc in docs
    ])

chaine_rag = (
    {
        "context":  RunnableLambda(
                        lambda q: formater_contexte(retriever.invoke(q)),
                        name="retrieval_formatting"  # nom visible dans LangSmith
                    ),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
).with_config({"run_name": "sentinel_rag_chain"})  # nom de la trace

print("✅ Pipeline optimisé construit")


# ── 3. SCÉNARIOS DE TEST TRACÉS ───────────────────────────────
print("\n" + "=" * 50)
print("3. EXÉCUTION AVEC TRACING LANGSMITH")
print("=" * 50)

scenarios = [
    # (question, description du scénario)
    ("Quel est le délai de remboursement en cas de fraude avérée ?",
     "Question directe — réponse attendue claire"),

    ("Quels pays nécessitent une validation manuelle systématique ?",
     "Question GAFI — problème identifié à l'étape 04"),

    ("Mon virement de 5000€ vers le Maroc est-il bloqué ?",
     "Question métier réelle — hors corpus exact"),

    ("Comment contester une transaction frauduleuse sur ma carte ?",
     "Question reformulée — test de robustesse"),

    ("Quel est le plafond pour payer sans authentification ?",
     "Question implicite — requiert inférence"),
]

for question, description in scenarios:
    print(f"\n{'─' * 50}")
    print(f"📋 Scénario : {description}")
    print(f"❓ {question}")

    reponse = chaine_rag.invoke(
        question,
        config={
            "metadata": {
                "scenario":  description,
                "version":   "v1.0",
                "chunk_size": 400,
                "k":          2,
            }
        }
    )
    print(f"🤖 {reponse}")

print(f"\n✅ {len(scenarios)} traces envoyées vers LangSmith")
print(f"🔗 Consultez : https://smith.langchain.com/projects/sentinel-rag")


# ── 4. CE QUE VOUS VOYEZ DANS LANGSMITH ───────────────────────
print("\n" + "=" * 50)
print("4. GUIDE DE LECTURE LANGSMITH")
print("=" * 50)

print("""
Dans l'interface LangSmith (smith.langchain.com) :

1. PROJECT : sentinel-rag
   → Liste de toutes vos traces

2. CLIQUEZ sur une trace
   → Vue arborescente de chaque étape :
     sentinel_rag_chain
     ├── retrieval_formatting  (retrieval + formatage)
     ├── ChatPromptTemplate    (construction du prompt)
     ├── sentinel-llm          (appel GPT-4o-mini)
     └── StrOutputParser       (parsing de la sortie)

3. POUR CHAQUE ÉTAPE vous voyez :
   → Input exact
   → Output exact
   → Latence en ms
   → Tokens consommés

4. COMPAREZ les traces
   → Question GAFI vs question fraude
   → Quels chunks ont été injectés dans le contexte ?
   → Pourquoi la question GAFI échoue ?
""")