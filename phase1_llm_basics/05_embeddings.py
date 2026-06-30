import os
import math
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI()


# ── FONCTION UTILITAIRE ───────────────────────────────────────

def get_embedding(texte: str) -> list[float]:
    """Convertit un texte en vecteur de 1536 dimensions."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texte,
    )
    return response.data[0].embedding


def similarite_cosinus(vecteur_a: list[float], vecteur_b: list[float]) -> float:
    """Mesure la similarité sémantique entre deux vecteurs. Résultat entre 0 et 1."""
    dot_product = sum(a * b for a, b in zip(vecteur_a, vecteur_b))
    norme_a = math.sqrt(sum(a ** 2 for a in vecteur_a))
    norme_b = math.sqrt(sum(b ** 2 for b in vecteur_b))
    return dot_product / (norme_a * norme_b)


# ── 1. ANATOMIE D'UN EMBEDDING ────────────────────────────────
print("=" * 50)
print("1. ANATOMIE D'UN EMBEDDING")
print("=" * 50)

vecteur = get_embedding("transaction suspecte au Nigeria")
print(f"Type       : {type(vecteur)}")
print(f"Dimensions : {len(vecteur)}")
print(f"Aperçu     : {vecteur[:5]} ...")
print(f"Min / Max  : {min(vecteur):.4f} / {max(vecteur):.4f}")


# ── 2. COMPARAISON SÉMANTIQUE ─────────────────────────────────
print("\n" + "=" * 50)
print("2. COMPARAISON SÉMANTIQUE")
print("=" * 50)

paires = [
    ("fraude bancaire",        "transaction suspecte"),      # très proches
    ("fraude bancaire",        "détection d'anomalie"),      # liés
    ("fraude bancaire",        "virement international"),    # domaine commun
    ("fraude bancaire",        "recette de cuisine"),        # sans rapport
    ("paiement par carte",     "règlement par CB"),          # synonymes métier
]

for texte_a, texte_b in paires:
    score = similarite_cosinus(get_embedding(texte_a), get_embedding(texte_b))
    barre = "█" * int(score * 20)
    print(f"{score:.4f} {barre}")
    print(f"         '{texte_a}' ↔ '{texte_b}'\n")


# ── 3. MINI MOTEUR DE RECHERCHE SÉMANTIQUE ────────────────────
print("=" * 50)
print("3. MINI MOTEUR DE RECHERCHE SÉMANTIQUE")
print("=" * 50)

# Base documentaire : transactions avec leurs descriptions
documents = [
    "Virement de 4850€ vers le Nigeria effectué à 03h17, 3e transaction en 10 minutes",
    "Paiement de 45€ chez Carrefour Paris à 14h30, comportement habituel du client",
    "Retrait de 500€ dans un distributeur à Bucarest, client jamais allé en Roumanie",
    "Achat en ligne de 120€ sur Amazon, adresse de livraison habituelle du client",
    "Tentative de paiement refusée de 9999€ vers un compte aux îles Caïmans",
    "Virement mensuel de 800€ vers le même bénéficiaire depuis 24 mois",
]

print("Indexation des documents...")
index = [(doc, get_embedding(doc)) for doc in documents]
print(f"{len(index)} documents indexés.\n")

def rechercher(requete: str, top_k: int = 3) -> list[tuple[float, str]]:
    """Retrouve les documents les plus pertinents pour une requête."""
    vecteur_requete = get_embedding(requete)
    scores = [
        (similarite_cosinus(vecteur_requete, vecteur_doc), doc)
        for doc, vecteur_doc in index
    ]
    return sorted(scores, reverse=True)[:top_k]

requetes = [
    "transaction inhabituelle en dehors de la France",
    "comportement normal du client",
    "montant excessif vers paradis fiscal",
]

for requete in requetes:
    print(f"🔍 Requête : '{requete}'")
    resultats = rechercher(requete, top_k=2)
    for rang, (score, doc) in enumerate(resultats, 1):
        print(f"  #{rang} ({score:.4f}) {doc}")
    print()


# ── 4. POURQUOI LE MODÈLE D'EMBEDDING COMPTE ─────────────────
print("=" * 50)
print("4. COMPARAISON DES MODÈLES D'EMBEDDING")
print("=" * 50)

texte_reference = "fraude par carte bancaire"
texte_similaire = "transaction frauduleuse par CB"

for modele in ["text-embedding-3-small", "text-embedding-3-large"]:
    vec_a = client.embeddings.create(model=modele, input=texte_reference).data[0].embedding
    vec_b = client.embeddings.create(model=modele, input=texte_similaire).data[0].embedding
    score = similarite_cosinus(vec_a, vec_b)
    dims  = len(vec_a)
    print(f"{modele}")
    print(f"  Dimensions : {dims} | Similarité : {score:.4f}\n")