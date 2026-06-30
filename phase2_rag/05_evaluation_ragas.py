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


# ── 1. RECONSTRUCTION DU PIPELINE RAG ────────────────────────
print("=" * 50)
print("1. RECONSTRUCTION DU PIPELINE")
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

embeddings    = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store  = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
    collection_name="regles_paiement",
)

retriever_dense = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4, "filter": {"chunk_id": {"$gt": 0}}},
)
retriever_bm25      = BM25Retriever.from_documents(chunks_utiles)
retriever_bm25.k    = 4

class HybridRetriever(BaseRetriever):
    retriever_dense: object = Field(...)
    retriever_bm25:  object = Field(...)
    reranker:        object = Field(...)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        docs_dense = self.retriever_dense.invoke(query)
        docs_bm25  = self.retriever_bm25.invoke(query)
        vus, fusionnes = set(), []
        for doc in docs_dense + docs_bm25:
            cle = doc.page_content[:80]
            if cle not in vus:
                vus.add(cle)
                fusionnes.append(doc)
        return self.reranker.compress_documents(fusionnes, query)

reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=3)
retriever_final = HybridRetriever(
    retriever_dense=retriever_dense,
    retriever_bm25=retriever_bm25,
    reranker=reranker,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

SYSTEM_PROMPT = """Tu es SENTINEL, un assistant expert en règles de paiement bancaire.
- Tu réponds UNIQUEMENT à partir du contexte fourni
- Si la réponse n'est pas dans le contexte, dis-le explicitement
- Tu es précis et factuel

CONTEXTE : {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def formater_contexte(docs: List[Document]) -> str:
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

chaine_rag = (
    {
        "context":  RunnableLambda(lambda q: formater_contexte(
                        retriever_final.invoke(q))),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Pipeline RAG reconstruit")


# ── 2. DATASET D'ÉVALUATION ───────────────────────────────────
print("\n" + "=" * 50)
print("2. DATASET D'ÉVALUATION")
print("=" * 50)

# Ground truth : questions + réponses de référence
# Couvre les 5 sections du document + 1 question hors contexte
eval_dataset = [
    {
        "question": "Quel est le délai de remboursement en cas de fraude avérée ?",
        "ground_truth": "En cas de fraude avérée, le remboursement est effectué sous 5 jours ouvrés après validation du dossier."
    },
    {
        "question": "Quel est le plafond de retrait journalier pour une carte premium ?",
        "ground_truth": "Le plafond de retrait pour une carte premium est de 1500€ par période de 24 heures."
    },
    {
        "question": "À partir de quel montant une transaction en ligne requiert-elle une authentification forte ?",
        "ground_truth": "Toute transaction en ligne supérieure à 30€ requiert une authentification forte (3DS2)."
    },
    {
        "question": "Quel est le délai maximum pour un virement SEPA ?",
        "ground_truth": "Les virements SEPA bénéficient d'un traitement prioritaire avec un délai maximal d'un jour ouvré."
    },
    {
        "question": "Combien de temps un client dispose-t-il pour contester une transaction ?",
        "ground_truth": "Un client dispose de 13 mois pour contester une transaction non reconnue."
    },
    {
        "question": "Quels pays nécessitent une validation manuelle systématique ?",
        "ground_truth": "Les pays classés à risque élevé selon la liste GAFI nécessitent une validation manuelle systématique."
    },
]

print(f"Dataset : {len(eval_dataset)} questions de référence")


# ── 3. GÉNÉRATION DES RÉPONSES ────────────────────────────────
print("\n" + "=" * 50)
print("3. GÉNÉRATION DES RÉPONSES")
print("=" * 50)

questions   = []
reponses    = []
contextes   = []
ground_truths = []

for item in eval_dataset:
    question = item["question"]
    print(f"  ⏳ {question[:60]}...")

    # Retrieval des contextes
    docs_retrieved = retriever_dense.invoke(question)
    contexte       = [doc.page_content for doc in docs_retrieved]

    # Génération de la réponse
    reponse = chaine_rag.invoke(question)

    questions.append(question)
    reponses.append(reponse)
    contextes.append(contexte)
    ground_truths.append(item["ground_truth"])

    print(f"  ✅ Réponse : {reponse[:80]}...")

print(f"\n{len(questions)} réponses générées")


# ── 4. MÉTRIQUES MANUELLES ────────────────────────────────────
print("\n" + "=" * 50)
print("4. ÉVALUATION MANUELLE")
print("=" * 50)

llm_eval = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def evaluer_faithfulness(question: str, reponse: str, contexte: list[str]) -> float:
    """
    Faithfulness : la réponse est-elle entièrement supportée par le contexte ?
    Le LLM juge chaque affirmation de la réponse — présente dans le contexte ou non.
    Score = nb affirmations supportées / nb affirmations totales
    """
    contexte_str = "\n".join(contexte)
    prompt_eval  = f"""Contexte :
{contexte_str}

Réponse à évaluer :
{reponse}

Chaque affirmation de la réponse est-elle entièrement supportée par le contexte ?
Réponds avec un score entre 0.0 et 1.0 uniquement (ex: 0.8).
1.0 = toutes les affirmations sont dans le contexte.
0.0 = aucune affirmation n'est dans le contexte.
Score :"""
    resultat = llm_eval.invoke(prompt_eval).content.strip()
    try:
        return float(resultat)
    except ValueError:
        return 0.0


def evaluer_answer_relevancy(question: str, reponse: str) -> float:
    """
    Answer Relevancy : la réponse répond-elle vraiment à la question ?
    Score = similarité cosinus entre la question et des questions générées
    à partir de la réponse.
    """
    prompt_eval = f"""Réponse :
{reponse}

Cette réponse répond-elle directement et complètement à la question suivante ?
Question : {question}

Score entre 0.0 et 1.0 uniquement.
1.0 = réponse parfaitement alignée avec la question.
0.0 = réponse hors sujet.
Score :"""
    resultat = llm_eval.invoke(prompt_eval).content.strip()
    try:
        return float(resultat)
    except ValueError:
        return 0.0


def evaluer_context_precision(question: str, contexte: list[str],
                               ground_truth: str) -> float:
    """
    Context Precision : les chunks retrievés sont-ils pertinents à la question ?
    Score = proportion de chunks utiles parmi ceux retrievés.
    """
    scores_chunks = []
    for chunk in contexte:
        prompt_eval = f"""Question : {question}
Réponse de référence : {ground_truth}
Chunk retrievé : {chunk}

Ce chunk est-il utile pour répondre à cette question ?
Score entre 0.0 et 1.0 uniquement.
1.0 = chunk directement utile.
0.0 = chunk inutile ou hors sujet.
Score :"""
        resultat = llm_eval.invoke(prompt_eval).content.strip()
        try:
            scores_chunks.append(float(resultat))
        except ValueError:
            scores_chunks.append(0.0)
    return sum(scores_chunks) / len(scores_chunks) if scores_chunks else 0.0


# ── 5. CALCUL ET AFFICHAGE DES RÉSULTATS ─────────────────────
print("\n" + "=" * 50)
print("5. RÉSULTATS")
print("=" * 50)

scores = []

for i, item in enumerate(eval_dataset):
    question    = item["question"]
    ground_truth = item["ground_truth"]
    reponse     = reponses[i]
    contexte    = contextes[i]

    print(f"\n⏳ Évaluation Q{i+1} : {question[:55]}...")

    f_score  = evaluer_faithfulness(question, reponse, contexte)
    ar_score = evaluer_answer_relevancy(question, reponse)
    cp_score = evaluer_context_precision(question, contexte, ground_truth)

    scores.append({
        "question":          question,
        "faithfulness":      f_score,
        "answer_relevancy":  ar_score,
        "context_precision": cp_score,
    })

    print(f"  Faithfulness      : {f_score:.3f}")
    print(f"  Answer Relevancy  : {ar_score:.3f}")
    print(f"  Context Precision : {cp_score:.3f}")

# Scores globaux
print("\n" + "=" * 50)
print("SCORES GLOBAUX")
print("=" * 50)

f_moy  = sum(s["faithfulness"]      for s in scores) / len(scores)
ar_moy = sum(s["answer_relevancy"]  for s in scores) / len(scores)
cp_moy = sum(s["context_precision"] for s in scores) / len(scores)

print(f"\n  Faithfulness      : {f_moy:.3f}  (anti-hallucination)")
print(f"  Answer Relevancy  : {ar_moy:.3f}  (pertinence réponse)")
print(f"  Context Precision : {cp_moy:.3f}  (qualité retrieval)")

print("\n⚠️  POINTS FAIBLES (score < 0.7) :")
alertes = 0
for i, s in enumerate(scores):
    for metrique in ["faithfulness", "answer_relevancy", "context_precision"]:
        if s[metrique] < 0.7:
            print(f"  Q{i+1} — {metrique} : {s[metrique]:.3f}")
            print(f"         → {s['question'][:60]}")
            alertes += 1

if alertes == 0:
    print("  Aucun point faible — excellent résultat !")