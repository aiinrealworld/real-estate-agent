from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import openai
from models.user_profile import UserProfile, validate_user_profile, normalize_user_profile, apply_defaults_to_profile
from models.property_recommendation import parse_chroma_results
import chromadb
import re
from dateparser import parse
from datetime import datetime, timedelta, time
import requests
import pytz
import json

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
    return recommendations

WORK_START = time(9, 0)
WORK_END = time(18, 0)
APPOINTMENT_DURATION = timedelta(hours=1)
BUFFER = timedelta(minutes=30)
AGENT_TIMEZONE = "America/Chicago"
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

@realtor_agent.tool
async def get_agent_availability(
    ctx: RunContext[UserProfile],
    date_time_preference: Optional[str] = None
) -> list[str]:

    tz = pytz.timezone(AGENT_TIMEZONE)
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

    # Fallback to "tomorrow at 00:00" if parsing fails
    if not parsed_datetime:
        parsed_datetime = datetime.now(tz) + timedelta(days=1)
        parsed_datetime = parsed_datetime.replace(hour=0, minute=0, second=0, microsecond=0)

    # Localize if naive
    if parsed_datetime.tzinfo is None:
        parsed_datetime = tz.localize(parsed_datetime)

    # Now compute free slots based on parsed datetime
    available_slots = compute_available_slots(parsed_datetime, AGENT_TIMEZONE)
    available_slots_formatted = format_slots_for_llm(available_slots, AGENT_TIMEZONE)
    print(f"formatted available slots for LLM {available_slots_formatted}")
    
    return available_slots_formatted

def compute_available_slots(
    parsed_datetime: datetime,
    timezone_str: str = "America/Chicago"
) -> list[str]:
    
    tz = pytz.timezone(timezone_str)
    date = parsed_datetime.astimezone(tz).date()  # just the date portion
    busy_slots = fetch_busy_slots_from_n8n(parsed_datetime)

    available_slots = []
    start_dt = tz.localize(datetime.combine(date, WORK_START))
    end_dt = tz.localize(datetime.combine(date, WORK_END)) - APPOINTMENT_DURATION

    current = start_dt
    while current <= end_dt:
        proposed_end = current + APPOINTMENT_DURATION
        conflict = any(
            (current < busy_end + BUFFER and proposed_end > busy_start - BUFFER)
            for busy_start, busy_end in busy_slots
        )
        if not conflict:
            available_slots.append(current.isoformat())
        current += timedelta(minutes=30)

    print(f"agents available slots {available_slots}")
    return available_slots

def fetch_busy_slots_from_n8n(start_datetime: datetime) -> list[tuple[datetime, datetime]]:
    end_datetime = start_datetime + timedelta(days=1)

    payload = {
        "mode": "get_busy_slots",
        "start": start_datetime.isoformat(),
        "end": end_datetime.isoformat()
    }

    response = requests.post(N8N_WEBHOOK_URL, json=payload)
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