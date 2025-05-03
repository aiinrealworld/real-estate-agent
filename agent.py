from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import openai
from models import UserProfile
import chromadb

from dotenv import load_dotenv
import os
from typing import Optional, List
from agent_config import SYSTEM_PROMPT

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_KEY")
llm_model = os.getenv("LLM_MODEL")

model = OpenAIModel(
    llm_model,
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key=openrouter_api_key
    )
)

# For embedding models
openai_key = os.getenv("OPENAI_API_KEY")

realtor_agent = Agent(
    model = model,
    system_prompt = SYSTEM_PROMPT,
    temperature=0.3,
    deps_type=UserProfile,
    output_type=str,
    instrument=True
)

from data.data_config import CHROMA_DB_LISTINGS, CHROMA_DB_PATH

# Setup Chroma
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(CHROMA_DB_LISTINGS)

@realtor_agent.tool
async def when_new_user_profile_info_received(
    ctx: RunContext[UserProfile],
    name: Optional[str] = None,
    phone: Optional[str] = None,
    location: Optional[str] = None,
    property_type: Optional[str] = None,
    budget: Optional[str] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
    must_haves: Optional[List[str]] = None,
    good_to_haves: Optional[List[str]] = None,
) -> str:

    """
        Updates the user profile with the provided information.
        Agent should call this whenever a new piece of information is received from the user.
    """

    profile = ctx.deps
    updated = False

    def update_field(field_name, value):
        nonlocal updated
        if value:
            setattr(profile, field_name, value)
            updated = True

    update_field("name", name)
    update_field("phone", phone)
    update_field("location", location)
    update_field("property_type", property_type)
    update_field("budget", budget)
    update_field("bedrooms", bedrooms)
    update_field("bathrooms", bathrooms)

    if must_haves:
        profile.must_haves.extend(must_haves)
        updated = True
    if good_to_haves:
        profile.good_to_haves.extend(good_to_haves)
        updated = True

    return "Profile updated." if updated else "No updates made."

@realtor_agent.tool
async def recommend_properties(ctx: RunContext[UserProfile]) -> dict:
    query = profile_to_text(ctx.deps)
    query_embedding = get_embedding(query)
    # Query Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    return results

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