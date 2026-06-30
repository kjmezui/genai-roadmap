from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")   # charge OPENAI_API_KEY depuis .env

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Tu es un assistant concis et précis."},
        {"role": "user", "content": "Explique-moi ce qu'est un embedding en 2 phrases."}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
print(f"\nTokens utilisés : {response.usage.total_tokens}")