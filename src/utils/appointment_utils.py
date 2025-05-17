# Standard library imports
from datetime import datetime, timedelta

# Third-party library imports
import requests

# Local application imports
from models.property_recommendation import PropertyRecommendation
from models.user_profile import UserProfile



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





