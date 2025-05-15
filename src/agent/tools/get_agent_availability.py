# Standard library imports
import re
from datetime import datetime, timedelta

# Third-party library imports
import pytz
from dateparser import parse
from pydantic_ai import RunContext
from typing import Optional

# Local application imports
from agent.agent_config import AgentDependencies
from models.user_profile import UserProfile, normalize_user_profile
from utils.time_utils import compute_available_slots, format_slots_for_llm
from agent.realtor_agent import realtor_agent


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

        # Only subtract 2 hours (buffer between appointments) if time is not exactly midnight (to avoid going to next day)
        if parsed_datetime.time() != datetime.min.time():
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