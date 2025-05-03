from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from models import UserProfile

from dotenv import load_dotenv
import os
from typing import Optional, List

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

system_prompt = """
    You are a helpful real estate voice agent. Be polite and helpful. Respond naturally as a human real estate agent would.

    Identify the missing information from the user profile summary and collect one missing information from the user at a time.

    You MUST use the `update_profile` tool to save each new piece of information. 

    Keep it friendly and conversational.\n\nOnce all fields are filled, respond to the user 

    \"Thank you! I now have everything I need to find you the perfect properties.\"
"""

realtor_agent = Agent(
    model = model,
    system_prompt = system_prompt,
    temperature=0,
    deps_type=UserProfile,
    output_type=str,
    instrument=True
)

@realtor_agent.tool
async def update_profile(
    ctx: RunContext[UserProfile],
    name: Optional[str] = None,
    phone: Optional[str] = None,
    location: Optional[str] = None,
    property_type: Optional[str] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
    must_haves: Optional[List[str]] = None,
    good_to_haves: Optional[List[str]] = None,
) -> str:

    """Updates the user profile with the provided information."""

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
    update_field("bedrooms", bedrooms)
    update_field("bathrooms", bathrooms)

    if must_haves:
        profile.must_haves.extend(must_haves)
        updated = True
    if good_to_haves:
        profile.good_to_haves.extend(good_to_haves)
        updated = True

    return "Profile updated." if updated else "No updates made."