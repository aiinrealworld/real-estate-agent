# Standard library imports
import os
import re
import json
from datetime import datetime, timedelta

# Third-party library imports
from dotenv import load_dotenv
import openai
import requests
import pytz
from dateparser import parse
from typing import Optional

# Pydantic AI imports
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Local application imports
from agent_config import SYSTEM_PROMPT, AgentDependencies
from models.user_profile import (
    UserProfile,
    validate_user_profile,
    normalize_user_profile,
    apply_defaults_to_profile,
)
from models.property_recommendation import (
    PropertyRecommendation,
    parse_chroma_results,
)
from models.agent_schedule_config import AgentScheduleConfig


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
    deps_type=AgentDependencies,
    output_type=str,
    instrument=True
)

@realtor_agent.tool
async def recommend_properties(
    ctx: RunContext[AgentDependencies],
    profile: UserProfile
) -> dict:

    print(f"user_profile in recommend_properties {profile}")

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

    chroma_client = ctx.deps.chroma_client
    chroma_db_listings = ctx.deps.chroma_db_listings

    listing_collection = chroma_client.get_collection(chroma_db_listings)
    # Query Chroma
    results = listing_collection.query(
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
    return recommendations

@realtor_agent.tool
async def get_agent_availability(
    ctx: RunContext[AgentDependencies],
    profile: UserProfile,
    date_time_preference: Optional[str] = None
) -> list[str]:

    normalized_user_profile = normalize_user_profile(profile)
    print(f"profile in get_agent_availability {normalized_user_profile}")

    agent_timezone = ctx.deps.agent_schedule_config.timezone
    n8n_webhook_url = ctx.deps.n8n_webhook_url

    tz = pytz.timezone(agent_timezone)
    parsed_datetime = None

    if date_time_preference:
        date_time_preference_clean = re.sub(r'\b(next|this|coming)\b', '', date_time_preference, flags=re.IGNORECASE)
        now = datetime.now(tz)
        print(f"now is set to {now}")
        parsed_datetime = parse(
            date_time_preference_clean, 
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": now
            }
        )
        print(f"preferred datetime '{date_time_preference_clean}' => {parsed_datetime}")

    # Apply buffer only if parsing succeeded
    if parsed_datetime:
        # Localize if needed
        if parsed_datetime.tzinfo is None:
            parsed_datetime = tz.localize(parsed_datetime)
        
        # Subtract 2 hours as scheduling buffer
        parsed_datetime -= timedelta(hours=2)
    # Fallback to "tomorrow at 00:00" if parsing fails
    else:
        parsed_datetime = datetime.now(tz) + timedelta(days=1)
        parsed_datetime = parsed_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
     

    # Localize if naive
    if parsed_datetime.tzinfo is None:
        parsed_datetime = tz.localize(parsed_datetime)

    agent_schedule_config = ctx.deps.agent_schedule_config
    # Now compute free slots based on parsed datetime
    available_slots = compute_available_slots(parsed_datetime, agent_schedule_config, n8n_webhook_url)
    available_slots_formatted = format_slots_for_llm(available_slots, agent_timezone)
    print(f"formatted available slots for LLM {available_slots_formatted}")
    
    return available_slots_formatted

def compute_available_slots(
    parsed_datetime: datetime,
    agent_schedule_config: AgentScheduleConfig,
    n8n_webhook_url: str
) -> list[str]:
    
    work_start = agent_schedule_config.work_start
    work_end = agent_schedule_config.work_end
    appointment_duration = agent_schedule_config.appointment_duration
    schedule_buffer = agent_schedule_config.schedule_buffer
    timezone = agent_schedule_config.timezone

    tz = pytz.timezone(timezone)
    date = parsed_datetime.astimezone(tz).date()  # just the date portion
    busy_slots = fetch_busy_slots_from_n8n(parsed_datetime, n8n_webhook_url)

    available_slots = []
    start_dt = tz.localize(datetime.combine(date, work_start))
    end_dt = tz.localize(datetime.combine(date, work_end)) - appointment_duration

    current = start_dt
    while current <= end_dt:
        proposed_end = current + appointment_duration
        conflict = any(
            (current < busy_end + schedule_buffer and proposed_end > busy_start - schedule_buffer)
            for busy_start, busy_end in busy_slots
        )
        if not conflict:
            available_slots.append(current.isoformat())
        current += timedelta(minutes=30)

    print(f"agents available slots {available_slots}")
    return available_slots

def fetch_busy_slots_from_n8n(
        start_datetime: datetime,
        n8n_webhook_url: str) -> list[tuple[datetime, datetime]]:
    end_datetime = start_datetime + timedelta(days=1)

    payload = {
        "mode": "get_busy_slots",
        "start": start_datetime.isoformat(),
        "end": end_datetime.isoformat()
    }

    response = requests.post(n8n_webhook_url, json=payload)
    response.raise_for_status()
    data = response.json()

    calendars = data.get("calendars", {})
    busy_slots = []
    for calendar_id, calendar_data in calendars.items():
        for item in calendar_data.get("busy", []):
            start = datetime.fromisoformat(item["start"])
            end = datetime.fromisoformat(item["end"])
            busy_slots.append((start, end))

    print(f"agent's schedule {busy_slots}")
    return busy_slots

def format_slots_for_llm(slots: list[str], tz_str="America/Chicago") -> str:
    tz = pytz.timezone(tz_str)

    # Current time in agent's timezone
    now = datetime.now(tz)
    current_time = now.strftime("%A, %B %d at %I:%M %p %Z")

    # Format each available slot
    formatted_slots = []
    for iso in slots:
        dt = datetime.fromisoformat(iso).astimezone(tz)
        formatted_time = dt.strftime("%A, %B %d at %I:%M %p").lstrip("0")
        formatted_slots.append(formatted_time)

    # Combine into a dictionary and return as JSON string
    output = {
        "current_time": current_time,
        "available_slots": formatted_slots
    }

    return json.dumps(output, indent=2)


@realtor_agent.tool
async def schedule_appointment(
        ctx: RunContext[AgentDependencies],
        profile: UserProfile,
        property: PropertyRecommendation,
        selected_date_time: str) -> list[str]:

    normalized_profile = normalize_user_profile(profile)
    print(f"user_profile in schedule_appointment {normalized_profile}")

    agent_timezone = ctx.deps.agent_schedule_config.timezone
    tz = pytz.timezone(agent_timezone)
    now = datetime.now(tz)

    # Try parsing human-friendly time
    start_dt = parse(
        selected_date_time,
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": now
        }
    )

    if not start_dt:
        return "Sorry, I couldn't understand the selected time. Please try again."

    # Localize if needed
    if start_dt.tzinfo is None:
        start_dt = tz.localize(start_dt)
    
    # Compute end time (1 hour after start)
    end_dt = start_dt + timedelta(hours=1)

    n8n_webhook_url = ctx.deps.n8n_webhook_url
    appt_response = send_appointment_to_n8n(normalized_profile, property, start_dt, end_dt, n8n_webhook_url)
    return appt_response

def send_appointment_to_n8n(
        profile: UserProfile,
        property: PropertyRecommendation,
        start_dt: datetime,
        end_dt: datetime,
        n8n_webhook_url: str
        ) -> str:

    # Construct event title and body
    title = f"Showing for {profile.name} ({property.address}, {property.city})"
    description = (
        f"Thank you for scheduling the showing with us. Here are the details: \n\n"
        f"Property ID: {property.listing_id}\n"
        f"User: {profile.name}\n"
        f"Phone: {profile.phone}\n"
        f"Property: {property.address}, {property.city}, {property.state}, {property.zip_code}\n"
        f"Appointment: {start_dt.strftime('%A, %B %d %Y at %I:%M %p')}"
    )

    payload = {
        "mode": "schedule_appointment",
        "listing_id": property.listing_id,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "title": title,
        "description": description,
        "user": {
            "name": profile.name,
            "phone": profile.phone
        }
    }

    print(f"schedule appt payload: {payload}")
    try:
        response = requests.post(n8n_webhook_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("confirmation_message", "Your appointment has been scheduled.")
    except Exception as e:
        print(f"[schedule_appointment] Failed to call n8n: {e}")
        return "There was an issue scheduling the appointment. Please try again later."

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