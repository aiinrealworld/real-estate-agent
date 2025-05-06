from typing import List, Optional
from pydantic import BaseModel

import re

class UserProfile(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    property_type: Optional[str] = None
    budget: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    must_haves: List[str] = []
    good_to_haves: List[str] = []

def normalize_user_profile(profile: UserProfile) -> UserProfile:
    return UserProfile(
        property_type=profile.property_type.strip().lower(),
        bedrooms=str(normalize_bedrooms(profile.bedrooms)),
        bathrooms=str(normalize_bathrooms(profile.bathrooms)),
        budget=str(normalize_price(profile.budget)),
        location=profile.location.strip().title(),
        must_haves=profile.must_haves or [],
        good_to_haves=profile.good_to_haves or []
    )

def normalize_price(price_input) -> int:
    if isinstance(price_input, (int, float)):
        return int(price_input)

    price_input = str(price_input).lower().replace(",", "").strip()
    matches = re.findall(r"[\d.]+", price_input)

    if not matches:
        return 0

    multiplier = 1
    if "k" in price_input:
        multiplier = 1_000
    elif "m" in price_input or "million" in price_input:
        multiplier = 1_000_000

    numbers = [float(match) * multiplier for match in matches]
    return int(max(numbers))

def normalize_number(input_val) -> float:
    if isinstance(input_val, (int, float)):
        return float(input_val)
    input_str = str(input_val).lower().replace(",", "").strip()
    match = re.search(r"[\d.]+", input_str)
    return float(match.group()) if match else 0.0


def normalize_bedrooms(input_str: str) -> int:
    return int(normalize_number(input_str))

def normalize_bathrooms(input_str: str) -> float:
    return normalize_number(input_str)

def normalize_sqft(input_str: str) -> int:
    return int(normalize_number(input_str))
