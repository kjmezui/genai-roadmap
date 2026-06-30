from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from pathlib import Path

DOCS_PATH = Path(__file__).parent / "documents"


# ── CHARGEMENT ────────────────────────────────────────────────
print("=" * 50)
print("CHARGEMENT DU DOCUMENT")
print("=" * 50)

loader = TextLoader(str(DOCS_PATH / "regles_paiement.txt"), encoding="utf-8")
documents = loader.load()

print(f"Nombre de documents chargés : {len(documents)}")
print(f"Taille totale               : {len(documents[0].page_content)} caractères")
print(f"\nExtrait :\n{documents[0].page_content[:200]}...")


# ── STRATÉGIE 1 : CHUNKING FIXE ───────────────────────────────
print("\n" + "=" * 50)
print("STRATÉGIE 1 : CHUNKING FIXE")
print("=" * 50)

splitter_fixe = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,       # taille cible en caractères
    chunk_overlap=50,     # overlap pour ne pas perdre le contexte aux frontières
    length_function=len,
)

chunks_fixes = splitter_fixe.split_documents(documents)

print(f"Nombre de chunks : {len(chunks_fixes)}")
print(f"\n--- Chunk #1 ---\n{chunks_fixes[0].page_content}")
print(f"\n--- Chunk #2 ---\n{chunks_fixes[1].page_content}")
print(f"\n--- Chunk #3 (début) ---\n{chunks_fixes[2].page_content[:150]}...")
#print(chunks_fixes[0].metadata)

# ── STRATÉGIE 2 : CHUNKING PAR SÉPARATEUR (recommandée) ───────
print("\n" + "=" * 50)
print("STRATÉGIE 2 : CHUNKING PAR SÉPARATEUR")
print("=" * 50)

splitter_recursif = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
    # ordre de priorité : paragraphe > ligne > phrase > mot > caractère
)

chunks_recursifs = splitter_recursif.split_documents(documents)

print(f"Nombre de chunks : {len(chunks_recursifs)}")

for i, chunk in enumerate(chunks_recursifs):
    print(f"\n--- Chunk #{i+1} ({len(chunk.page_content)} chars) ---")
    print(chunk.page_content)
    print(f"Métadonnées : {chunk.metadata}")


# ── ANALYSE COMPARATIVE ───────────────────────────────────────
print("\n" + "=" * 50)
print("ANALYSE COMPARATIVE")
print("=" * 50)

tailles_fixes    = [len(c.page_content) for c in chunks_fixes]
tailles_recursif = [len(c.page_content) for c in chunks_recursifs]

print(f"Chunking fixe      : {len(chunks_fixes)} chunks")
print(f"  Taille min/moy/max : {min(tailles_fixes)} / "
      f"{sum(tailles_fixes)//len(tailles_fixes)} / {max(tailles_fixes)} chars")

print(f"\nChunking récursif  : {len(chunks_recursifs)} chunks")
print(f"  Taille min/moy/max : {min(tailles_recursif)} / "
      f"{sum(tailles_recursif)//len(tailles_recursif)} / {max(tailles_recursif)} chars")


# ── OVERLAP : VISUALISATION ───────────────────────────────────
print("\n" + "=" * 50)
print("VISUALISATION DE L'OVERLAP")
print("=" * 50)

print("Fin du chunk #1 :")
print(f"  ...{chunks_recursifs[0].page_content[-80:]}")
print("\nDébut du chunk #2 :")
print(f"  {chunks_recursifs[1].page_content[:80]}...")
print("\n→ Les caractères en commun = overlap (contexte préservé)")

fin_chunk1   = chunks_recursifs[0].page_content[-80:]
debut_chunk2 = chunks_recursifs[1].page_content[:80]

# Trouver les mots communs
mots_fin    = set(fin_chunk1.split())
mots_debut  = set(debut_chunk2.split())
communs     = mots_fin & mots_debut

print("Mots communs entre chunk #1 et chunk #2 :")
print(communs)