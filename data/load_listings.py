import json
import openai
import chromadb
from time import sleep

from dotenv import load_dotenv
import os
from data_config import CHROMA_DB_LISTINGS, CHROMA_DB_PATH, LISTINGS_DATASET

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# Text generation function
def generate_property_text(property_data: dict) -> str:
    parts = []
    parts.append(f"{property_data.get('property_type', 'Property')} for sale")
    if 'bedrooms' in property_data:
        parts.append(f"with {property_data['bedrooms']} bedroom{'s' if property_data['bedrooms'] != 1 else ''}")
    if 'bathrooms' in property_data:
        parts.append(f"and {property_data['bathrooms']} bathroom{'s' if property_data['bathrooms'] != 1 else ''}")
    if 'neighborhood' in property_data:
        parts.append(f"in {property_data['neighborhood']},")
    if 'city' in property_data and 'state' in property_data:
        parts.append(f"{property_data['city']}, {property_data['state']}.")
    if 'price' in property_data:
        parts.append(f"Priced at ${property_data['price']:,}")
    if 'square_feet' in property_data:
        parts.append(f"with {property_data['square_feet']:,} square feet of living space.")
    if 'lot_size' in property_data:
        parts.append(f"The lot size is {property_data['lot_size']} acres.")
    if 'address' in property_data:
        parts.append(f"Located at {property_data['address']}.")
    if 'year_built' in property_data:
        parts.append(f"Built in {property_data['year_built']}.")
    if 'mls_status' in property_data:
        parts.append(f"MLS Status: {property_data['mls_status']}.")
    if 'days_on_market' in property_data:
        parts.append(f"On the market for {property_data['days_on_market']} days.")
    if 'description' in property_data:
        parts.append(property_data['description'])

    return ' '.join(parts)

# OpenAI embedding function
def get_openai_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

# Load listings
with open(LISTINGS_DATASET, "r") as f:
    listings = json.load(f)

# Init Chroma
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(CHROMA_DB_LISTINGS)

# Batch insert
BATCH_SIZE = 100
for i in range(0, len(listings), BATCH_SIZE):
    batch = listings[i:i + BATCH_SIZE]
    documents = [generate_property_text(item) for item in batch]
    ids = [item["listing_id"] for item in batch]
    metadatas = [{k: v for k, v in item.items()} for item in batch]

    try:
        embeddings = get_openai_embeddings(documents)
    except openai.RateLimitError:
        print("⚠️ Rate limit hit. Waiting and retrying...")
        sleep(20)
        embeddings = get_openai_embeddings(documents)

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    print(f"✅ Inserted {i + len(batch)} of {len(listings)} listings.")
