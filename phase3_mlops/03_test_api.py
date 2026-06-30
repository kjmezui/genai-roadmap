import httpx
import time

BASE_URL = "http://127.0.0.1:8000"


print("=" * 50)
print("1. TEST DE SANTÉ")
print("=" * 50)

response = httpx.get(f"{BASE_URL}/health")
print(f"Status : {response.status_code}")
print(response.json())


print("\n" + "=" * 50)
print("2. QUESTIONS AU SERVICE")
print("=" * 50)

questions = [
    "Quel est le délai de remboursement en cas de fraude avérée ?",
    "Quels pays nécessitent une validation manuelle ?",
    "Quelle est la couleur du logo de la banque ?",  # hors contexte
]

for question in questions:
    print(f"\n❓ {question}")
    debut = time.time()
    response = httpx.post(
        f"{BASE_URL}/ask",
        json={"question": question},
        timeout=30.0,
    )
    duree = (time.time() - debut) * 1000

    if response.status_code == 200:
        data = response.json()
        print(f"🤖 {data['reponse']}")
        print(f"⏱️  Latence serveur : {data['latence_ms']}ms "
              f"(round-trip total : {duree:.0f}ms)")
    else:
        print(f"❌ Erreur {response.status_code} : {response.text}")


print("\n" + "=" * 50)
print("3. TEST DE VALIDATION (question trop courte)")
print("=" * 50)

response = httpx.post(f"{BASE_URL}/ask", json={"question": "Oui"})
print(f"Status : {response.status_code}")
print(response.json())


print("\n" + "=" * 50)
print("4. MÉTRIQUES FINALES")
print("=" * 50)

response = httpx.get(f"{BASE_URL}/metrics")
print(response.json())