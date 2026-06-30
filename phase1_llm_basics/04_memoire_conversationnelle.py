from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI()

SYSTEM = """Tu es SENTINEL, assistant expert en analyse de transactions de paiement.
Tu te souviens du contexte client établi en début de conversation.
Tu réponds de façon concise et factuelle."""


def appeler(messages: list) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ── 1. MÉMOIRE COMPLÈTE ───────────────────────────────────────
print("=" * 50)
print("1. MÉMOIRE COMPLÈTE")
print("=" * 50)

historique = [{"role": "system", "content": SYSTEM}]

conversations = [
    "Le client s'appelle Marc Dupont, compte premium, domicilié en France.",
    "Il vient d'effectuer une transaction de 4850€ au Nigeria à 03h17.",
    "C'est sa 3e transaction en 10 minutes. Quel est ton verdict ?",
    "Quel est le prénom du client que tu analyses ?",  # test de mémoire
]

for message in conversations:
    print(f"\n👤 User : {message}")
    historique.append({"role": "user", "content": message})
    reponse = appeler(historique)
    historique.append({"role": "assistant", "content": reponse})
    print(f"🤖 SENTINEL : {reponse}")

print(f"\n📊 Nombre de messages dans l'historique : {len(historique)}")


# ── 2. MÉMOIRE GLISSANTE ──────────────────────────────────────
print("\n" + "=" * 50)
print("2. MÉMOIRE GLISSANTE (fenêtre = 3 tours)")
print("=" * 50)

def appeler_avec_fenetre(historique: list, system: str, fenetre: int = 3) -> str:
    """Ne garde que les N derniers tours + le system prompt."""
    messages = [{"role": "system", "content": system}]
    messages += historique[-(fenetre * 2):]  # *2 car 1 tour = user + assistant
    return appeler(messages)

historique_glissant = []

for message in conversations:
    print(f"\n👤 User : {message}")
    historique_glissant.append({"role": "user", "content": message})
    reponse = appeler_avec_fenetre(historique_glissant, SYSTEM, fenetre=3)
    historique_glissant.append({"role": "assistant", "content": reponse})
    print(f"🤖 SENTINEL : {reponse}")

print(f"\n📊 Taille historique total : {len(historique_glissant)} messages")
print(f"📊 Taille fenêtre envoyée : 6 messages max (3 tours)")


# ── 3. MÉMOIRE RÉSUMÉE ────────────────────────────────────────
print("\n" + "=" * 50)
print("3. MÉMOIRE RÉSUMÉE")
print("=" * 50)

def resumer_historique(historique: list) -> str:
    """Résume l'historique en un bloc compact."""
    contenu = "\n".join([f"{m['role']}: {m['content']}" for m in historique])
    prompt = f"Résume ce contexte de conversation en 3 phrases max, en conservant les faits clés :\n\n{contenu}"
    return appeler([{"role": "user", "content": prompt}])

historique_complet = []
resume = ""
SEUIL_RESUME = 4  # on résume tous les 4 messages

for i, message in enumerate(conversations):
    print(f"\n👤 User : {message}")

    # Construction des messages avec résumé si disponible
    messages = [{"role": "system", "content": SYSTEM}]
    if resume:
        messages.append({"role": "system", "content": f"Contexte résumé : {resume}"})
    messages.append({"role": "user", "content": message})

    reponse = appeler(messages)
    historique_complet.append({"role": "user", "content": message})
    historique_complet.append({"role": "assistant", "content": reponse})
    print(f"🤖 SENTINEL : {reponse}")

    # Résumé périodique
    if len(historique_complet) >= SEUIL_RESUME and len(historique_complet) % SEUIL_RESUME == 0:
        resume = resumer_historique(historique_complet)
        print(f"\n📝 Résumé généré : {resume}")
        historique_complet = []  # on repart avec un historique vide