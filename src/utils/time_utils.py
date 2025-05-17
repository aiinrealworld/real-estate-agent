# Standard library imports
from datetime import datetime, timedelta
import json

# Third-party library imports
import pytz

# Local application imports
from models.agent_schedule_config import AgentScheduleConfig
from utils.appointment_utils import fetch_busy_slots_from_n8n


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