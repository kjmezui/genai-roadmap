from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI()

def appeler_modele(system: str, user: str, temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ── 1. ZERO-SHOT ──────────────────────────────────────────────
print("=" * 50)
print("1. ZERO-SHOT")
print("=" * 50)

system_zero = "Tu es un expert en détection de fraude bancaire."
user_zero = """Analyse cette transaction et dis si elle est suspecte :
Montant: 4 850€ | Pays: Nigeria | Heure: 03h17 | Fréquence: 3e transaction en 10 minutes"""

print(appeler_modele(system_zero, user_zero))


# ── 2. FEW-SHOT ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. FEW-SHOT")
print("=" * 50)

system_few = """Tu es un expert en détection de fraude bancaire.
Tu analyses des transactions et retournes UNIQUEMENT ce format JSON :
{"decision": "SUSPECTE|NORMALE", "score_risque": 0-100, "motif": "raison courte"}"""

user_few = """Exemples :
Transaction: 50€ | France | 14h30 | 1ère du jour → {"decision": "NORMALE", "score_risque": 5, "motif": "transaction courante"}
Transaction: 3200€ | Roumanie | 02h45 | 5e en 1h → {"decision": "SUSPECTE", "score_risque": 92, "motif": "montant élevé, heure atypique, fréquence anormale"}

Maintenant analyse :
Transaction: 4850€ | Nigeria | 03h17 | 3e en 10 minutes"""

print(appeler_modele(system_few, user_few))


# ── 3. CHAIN-OF-THOUGHT ───────────────────────────────────────
print("\n" + "=" * 50)
print("3. CHAIN-OF-THOUGHT")
print("=" * 50)

system_cot = "Tu es un expert en détection de fraude bancaire."
user_cot = """Analyse cette transaction étape par étape :

Étape 1 : Évalue le montant (normal pour ce type de compte ?)
Étape 2 : Évalue la géographie (pays habituel du client ?)
Étape 3 : Évalue l'horaire (heure typique ?)
Étape 4 : Évalue la fréquence (comportement habituel ?)
Étape 5 : Synthèse et décision finale

Transaction: 4850€ | Nigeria | 03h17 | 3e transaction en 10 minutes"""

print(appeler_modele(system_cot, user_cot))


# ── 4. SYSTEM PROMPT STRUCTURÉ ────────────────────────────────
print("\n" + "=" * 50)
print("4. SYSTEM PROMPT STRUCTURÉ")
print("=" * 50)

system_struct = """Tu es SENTINEL, un système expert en détection de fraude pour une banque française.

RÈGLES STRICTES :
- Tu analyses UNIQUEMENT des transactions de paiement
- Tu réponds TOUJOURS dans ce format :
  VERDICT: [BLOQUER / SURVEILLER / AUTORISER]
  CONFIANCE: [pourcentage]
  FACTEURS DE RISQUE: [liste des signaux détectés]
  ACTION RECOMMANDÉE: [instruction précise pour l'opérateur]
- Tu ne donnes jamais d'avis sur autre chose que la fraude
- Ton ton est factuel, sans émotion"""

user_struct = "Transaction: 4850€ | Nigeria | 03h17 | 3e en 10 minutes | Client habituel France"

print(appeler_modele(system_struct, user_struct))