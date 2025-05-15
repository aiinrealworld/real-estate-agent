# Standard library imports
from datetime import datetime, timedelta

# Third-party library imports
import pytz
from dateparser import parse
from pydantic_ai import RunContext

# Local application imports
from utils.appointment_utils import send_appointment_to_n8n
from agent.agent_config import AgentDependencies
from agent.realtor_agent import realtor_agent
from models.property_recommendation import PropertyRecommendation
from models.user_profile import UserProfile, normalize_user_profile


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