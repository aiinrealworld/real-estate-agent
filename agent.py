from pydantic_ai import Agent, RunContext
from models import UserProfile

from dotenv import load_dotenv
import os
from typing import Optional, List

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL", "openai:gpt-4o")
system_prompt = os.getenv("SYSTEM_PROMPT")

realtor_agent = Agent(
    model = llm_model,
    system_prompt = system_prompt,
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