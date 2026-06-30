import mlflow
import mlflow.artifacts
import json
import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from typing import List
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CHROMA_PATH = Path(__file__).resolve().parents[1] / "phase2_rag" / "chroma_db"
DOCS_PATH   = Path(__file__).resolve().parents[1] / "phase2_rag" / "documents"


# ── FONCTIONS UTILITAIRES ─────────────────────────────────────

def construire_pipeline(chunk_size: int, chunk_overlap: int, k: int,
                         temperature: float) -> tuple:
    """Construit le pipeline RAG avec les paramètres donnés."""

    loader = TextLoader(str(DOCS_PATH / "regles_paiement.txt"), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
        search_kwargs={"k": k, "filter": {"chunk_id": {"$gt": 0}}},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

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

    chaine = (
        {
            "context":  RunnableLambda(lambda q: formater_contexte(
                            retriever.invoke(q))),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chaine, chunks


def evaluer_pipeline(retriever, chaine, eval_dataset: list) -> dict:
    """Évalue le pipeline et retourne les scores moyens."""

    llm_eval = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def faithfulness(reponse: str, contexte: list[str]) -> float:
        ctx = "\n---\n".join(contexte)
        p   = f"Contexte :\n{ctx}\n\nRéponse :\n{reponse}\n\nToutes les affirmations sont-elles dans le contexte ? Score 0.0-1.0 uniquement :"
        try:    return float(llm_eval.invoke(p).content.strip())
        except: return 0.0

    def answer_relevancy(question: str, reponse: str) -> float:
        p = f"Question : {question}\nRéponse : {reponse}\n\nLa réponse répond-elle directement ? Score 0.0-1.0 uniquement :"
        try:    return float(llm_eval.invoke(p).content.strip())
        except: return 0.0

    def context_precision(question: str, contexte: list[str],
                          ground_truth: str) -> float:
        scores = []
        for chunk in contexte:
            p = f"Question : {question}\nRéférence : {ground_truth}\nChunk : {chunk}\n\nCe chunk est-il utile ? Score 0.0-1.0 uniquement :"
            try:    scores.append(float(llm_eval.invoke(p).content.strip()))
            except: scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    f_scores, ar_scores, cp_scores = [], [], []
    latences = []

    for item in eval_dataset:
        question     = item["question"]
        ground_truth = item["ground_truth"]

        # Mesure de latence
        debut = time.time()
        docs  = retriever.invoke(question)
        reponse = chaine.invoke(question)
        latences.append(time.time() - debut)

        contexte = [d.page_content for d in docs]

        f_scores.append(faithfulness(reponse, contexte))
        ar_scores.append(answer_relevancy(question, reponse))
        cp_scores.append(context_precision(question, contexte, ground_truth))

    return {
        "faithfulness":      sum(f_scores)  / len(f_scores),
        "answer_relevancy":  sum(ar_scores) / len(ar_scores),
        "context_precision": sum(cp_scores) / len(cp_scores),
        "latence_moyenne":   sum(latences)  / len(latences),
    }


# ── DATASET D'ÉVALUATION ──────────────────────────────────────

eval_dataset = [
    {"question": "Quel est le délai de remboursement en cas de fraude avérée ?",
     "ground_truth": "En cas de fraude avérée, le remboursement est effectué sous 5 jours ouvrés après validation du dossier."},
    {"question": "Quel est le plafond de retrait journalier pour une carte premium ?",
     "ground_truth": "Le plafond de retrait pour une carte premium est de 1500€ par période de 24 heures."},
    {"question": "À partir de quel montant une transaction en ligne requiert-elle une authentification forte ?",
     "ground_truth": "Toute transaction en ligne supérieure à 30€ requiert une authentification forte (3DS2)."},
    {"question": "Quel est le délai maximum pour un virement SEPA ?",
     "ground_truth": "Les virements SEPA bénéficient d'un traitement prioritaire avec un délai maximal d'un jour ouvré."},
    {"question": "Combien de temps un client dispose-t-il pour contester une transaction ?",
     "ground_truth": "Un client dispose de 13 mois pour contester une transaction non reconnue."},
    {"question": "Quels pays nécessitent une validation manuelle systématique ?",
     "ground_truth": "Les pays classés à risque élevé selon la liste GAFI nécessitent une validation manuelle systématique."},
]


# ── EXPÉRIENCES MLFLOW ────────────────────────────────────────

print("=" * 50)
print("EXPÉRIENCES MLFLOW")
print("=" * 50)

mlflow.set_tracking_uri("sqlite:///phase3_mlops/mlflow.db")
mlflow.set_experiment("RAG_chunking_experiments")

# Configurations à comparer
configurations = [
    {"chunk_size": 400, "chunk_overlap": 80,  "k": 4, "temperature": 0.1},
    {"chunk_size": 400, "chunk_overlap": 80,  "k": 2, "temperature": 0.1},
    {"chunk_size": 200, "chunk_overlap": 40,  "k": 4, "temperature": 0.1},
]

for config in configurations:
    run_name = (f"chunk{config['chunk_size']}_"
                f"overlap{config['chunk_overlap']}_"
                f"k{config['k']}")

    print(f"\n⏳ Run : {run_name}")

    with mlflow.start_run(run_name=run_name):

        # Log des paramètres
        mlflow.log_params(config)
        mlflow.log_param("embedding_model", "text-embedding-3-small")
        mlflow.log_param("llm_model",       "gpt-4o-mini")

        # Construction et évaluation
        retriever, chaine, chunks = construire_pipeline(**config)
        scores = evaluer_pipeline(retriever, chaine, eval_dataset)

        # Log des métriques
        mlflow.log_metrics(scores)
        mlflow.log_metric("nb_chunks", len(chunks))

        # Log du prompt template comme artifact
        prompt_txt = """Tu es SENTINEL, un assistant expert en règles de paiement bancaire.
- Tu réponds UNIQUEMENT à partir du contexte fourni
- Si la réponse n'est pas dans le contexte, dis-le explicitement
- Tu es précis et factuel"""

        prompt_path = Path("phase3_mlops/prompt_template.txt")
        prompt_path.write_text(prompt_txt)
        mlflow.log_artifact(str(prompt_path))

        print(f"  ✅ Faithfulness      : {scores['faithfulness']:.3f}")
        print(f"  ✅ Answer Relevancy  : {scores['answer_relevancy']:.3f}")
        print(f"  ✅ Context Precision : {scores['context_precision']:.3f}")
        print(f"  ✅ Latence moyenne   : {scores['latence_moyenne']:.2f}s")
        print(f"  ✅ Nb chunks         : {len(chunks)}")

print("\n" + "=" * 50)
print("COMPARAISON DES RUNS")
print("=" * 50)

runs = mlflow.search_runs(
    experiment_names=["RAG_chunking_experiments"],
    order_by=["metrics.context_precision DESC"],
)

colonnes = ["run_id", "params.chunk_size", "params.k",
            "metrics.faithfulness", "metrics.answer_relevancy",
            "metrics.context_precision", "metrics.latence_moyenne"]

print(runs[colonnes].to_string(index=False))

meilleur = runs.iloc[0]
print(f"\n🏆 Meilleure configuration :")
print(f"   chunk_size       = {meilleur['params.chunk_size']}")
print(f"   k                = {meilleur['params.k']}")
print(f"   context_precision = {meilleur['metrics.context_precision']:.3f}")
# Affichage directement des données sur le terminal - sans UI.
print("\n" + "=" * 50)
print("RAPPORT MLFLOW DÉTAILLÉ")
print("=" * 50)

runs = mlflow.search_runs(
    experiment_names=["RAG_chunking_experiments"],
    order_by=["metrics.context_precision DESC"],
)

for _, run in runs.iterrows():
    print(f"""
┌─────────────────────────────────────────┐
│ Run : chunk{run['params.chunk_size']}_k{run['params.k']}
├─────────────────────────────────────────┤
│ Faithfulness      : {run['metrics.faithfulness']:.3f}
│ Answer Relevancy  : {run['metrics.answer_relevancy']:.3f}
│ Context Precision : {run['metrics.context_precision']:.3f}
│ Latence moyenne   : {run['metrics.latence_moyenne']:.2f}s
│ Nb chunks         : {run['metrics.nb_chunks']:.0f}
└─────────────────────────────────────────┘""")