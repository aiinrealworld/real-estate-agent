import openai
import chromadb
from chromadb.config import Settings

from dotenv import load_dotenv
import os

from data_config import CHROMA_DB_LISTINGS, CHROMA_DB_PATH

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# Setup Chroma
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(CHROMA_DB_LISTINGS)

# Function to embed a query and perform similarity search
def search_listings(query: str, n_results: int = 3):
    response = openai.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\n🔍 Query: {query}")
    for i in range(n_results):
        print(f"\n--- Result {i+1} ---")
        print(f"📄 Document: {results['documents'][0][i][:300]}...")
        print(f"📎 Metadata: {results['metadatas'][0][i]}")
        print(f"📏 Distance: {results['distances'][0][i]:.4f}")

# 🔁 Test with sample queries
sample_queries = [
    "townhome, 3 bedroom(s), 2 bathroom(s), 450000 budget, in Naperville, Must haves: garage, backyard, Good to haves: near train station, finished basement"
]

"""
    "condo, 2 bedroom(s), 2 bathroom(s), 350000 budget, in downtown Chicago, Must haves: balcony, in-unit laundry, Good to haves: gym, parking spot",
    "apartment, 1 bedroom(s), 1 bathroom(s), 200000 budget, in Evanston, Must haves: elevator, pet friendly, Good to haves: doorman, dishwasher",
    "single-family home, 5 bedroom(s), 4 bathroom(s), 1200000 budget, in Lincoln Park, Must haves: pool, gourmet kitchen, Good to haves: home theater, wine cellar",
    "townhome, 2 bedroom(s), 1 bathroom(s), 300000 budget, in Schaumburg, Must haves: low HOA fees, safe neighborhood, Good to haves: nearby schools, guest parking",
    "condo, 2 bedroom(s), 2 bathroom(s), 400000 budget, in Oak Park, Must haves: no stairs, walk-in shower, Good to haves: community activities, nearby golf course",
    "single-family home, 3 bedroom(s), 2 bathroom(s), 500000 budget, in Naperville, Must haves: home office space, high-speed internet, Good to haves: coffee shop nearby, park access",
    "single-family home, 4 bedroom(s), 3 bathroom(s), 650000 budget, in Hinsdale, Must haves: top school district, fenced yard, Good to haves: playground nearby, finished basement"

"""

for q in sample_queries:
    search_listings(q, n_results=2)
