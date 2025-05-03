import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Create persistent Chroma client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Get or create collection
collection = chroma_client.get_or_create_collection("properties")

# Function to embed using OpenAI
def get_embedding(text: str):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"  # Choose your embedding model
    )
    return response.data[0].embedding

# Sample property listings
properties = [
    {
        "id": "prop1",
        "description": "Modern 2 bedroom condo in Lincoln Park, 1.5 bathrooms, includes gym access. $300,000."
    },
    {
        "id": "prop2",
        "description": "Cozy 3-bedroom family home in Naperville, 2 baths, large backyard, $500,000"
    },
    {
        "id": "prop3",
        "description": "Studio apartment in downtown Chicago with lake view. $215,000"
    },
]

# Add to ChromaDB
for prop in properties:
    embedding = get_embedding(prop["description"])
    collection.add(
        documents=[prop["description"]],
        embeddings=[embedding],
        ids=[prop["id"]]
    )

print("Properties indexed and saved.")
