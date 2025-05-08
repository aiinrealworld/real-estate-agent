from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import openai
from models.user_profile import UserProfile, validate_user_profile, normalize_user_profile, apply_defaults_to_profile
from models.property_recommendation import PropertyRecommendation, parse_chroma_results
import chromadb

from dotenv import load_dotenv
import os
from typing import Optional, List
from agent_config import SYSTEM_PROMPT


load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_KEY")
llm_model = os.getenv("OPEN_ROUTER_LLM_MODEL")

model = OpenAIModel(
    llm_model,
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key=openrouter_api_key
    )
)

# Overriding to use OpenAI directly
model = os.getenv("OPENAI_LLM_MODEL")
openai_key = os.getenv("OPENAI_API_KEY")

realtor_agent = Agent(
    model = model,
    system_prompt = SYSTEM_PROMPT,
    temperature=0.3,
    deps_type=UserProfile,
    output_type=str,
    instrument=True
)

from data.data_config import CHROMA_DB_LISTINGS

# Setup Chroma
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(CHROMA_DB_LISTINGS)

@realtor_agent.tool
async def recommend_properties(
    ctx: RunContext[UserProfile],
    name: Optional[str] = None,
    phone: Optional[str] = None,
    buyOrRent: Optional[str] = None,
    location: Optional[str] = None,
    property_type: Optional[str] = None,
    sqft: Optional[str] = None,
    budget: Optional[str] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
    must_haves: Optional[List[str]] = None,
    good_to_haves: Optional[List[str]] = None
) -> dict:

    profile = ctx.deps

    setattr(profile, "name", name)
    setattr(profile, "phone", phone)
    setattr(profile, "buyOrRent", buyOrRent)
    setattr(profile, "location", location)
    setattr(profile, "property_type", property_type)
    setattr(profile, "sqft", sqft)
    setattr(profile, "budget", budget)
    setattr(profile, "bedrooms", bedrooms)
    setattr(profile, "bathrooms", bathrooms)

    if must_haves:
        profile.must_haves.extend(must_haves)
    if good_to_haves:
        profile.good_to_haves.extend(good_to_haves)

    validation_errors = validate_user_profile(profile)
    if validation_errors:
        return validation_errors

    user_profile_with_defaults = apply_defaults_to_profile(profile)
    normalized_user_profile = normalize_user_profile(user_profile_with_defaults)
    print(f"normalized profile: {normalized_user_profile}")

    query = profile_to_text(normalized_user_profile)
    query_embedding = get_embedding(query)
    
    price_tolerance = 50000
    sqft_tolerance = 300

    # Query Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where = {
            "$and": [
                {"city": {"$eq": normalized_user_profile.location}},
                {"property_type": {"$eq": normalized_user_profile.property_type}},
                {"square_feet": {"$gte": int(normalized_user_profile.sqft) - sqft_tolerance}},
                {"price": {"$gte": int(normalized_user_profile.budget) - price_tolerance}},
                {"price": {"$lte": int(normalized_user_profile.budget) + price_tolerance}},
                {"bedrooms": {"$gte": int(normalized_user_profile.bedrooms)}},
                {"bathrooms": {"$gte": int(normalized_user_profile.bathrooms)}}
            ]
        }

    )
    recommendations = parse_chroma_results(results)
    print(f"recommended results {recommendations}")
    return recommendations


@realtor_agent.tool
async def get_available_slots(
        ctx: RunContext[UserProfile],
        listing_id: str) -> list[str]:
    
    print(f"property_id: {listing_id}")
    AVAILABLE_SLOTS = {
        "CH1954": ["Wednesday at 3 PM", "Thursday at 10 AM"],
        "CH1970": ["Friday at 2 PM", "Saturday at 11 AM"],
        "CH1893": ["Monday at 5 PM", "Tuesday at 1 PM"],
    }
    return AVAILABLE_SLOTS.get(listing_id, ["No slots available"])


@realtor_agent.tool
async def schedule_appointment(
        ctx: RunContext[UserProfile],
        selected_date_time: str) -> list[str]:
    
    profile = ctx.deps
    return f"Appointment confirmed for {selected_date_time}. A text confirmation has been sent to {profile.phone}."


def profile_to_text(profile: UserProfile) -> str:
    return (
        f"{profile.property_type or ''}, "
        f"{profile.bedrooms or ''} bedroom(s), "
        f"{profile.bathrooms or ''} bathroom(s), "
        f"{profile.budget or ''} budget, "
        f"in {profile.location or ''}, "
        f"Must haves: {', '.join(profile.must_haves)}, "
        f"Good to haves: {', '.join(profile.good_to_haves)}"
    )

# Function to embed using OpenAI
def get_embedding(text: str):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding