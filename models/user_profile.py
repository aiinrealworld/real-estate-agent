from typing import List, Optional
from pydantic import BaseModel

import re

class UserProfile(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    buyOrRent: Optional[str] = None
    location: Optional[str] = None
    property_type: Optional[str] = None
    sqft: Optional[str] = None
    budget: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    must_haves: List[str] = []
    good_to_haves: List[str] = []

def validate_user_profile(profile: UserProfile) -> List[str]:
    errors = []

    if not profile.name or not profile.name.strip():
        errors.append("Name is required and cannot be empty.")

    if not validate_phone_number(profile.phone):
        errors.append("Phone number is required and must have 10 to 15 digits.")

    if not profile.location or not profile.location.strip():
        errors.append("Location is required and cannot be empty.")

    if not profile.budget or not str(profile.budget).strip().isdigit():
        errors.append("Budget is required and must be a numeric value.")

    valid_property_types = {"Multi-Family", "Condo", "Single Family", "Townhouse"}
    if not profile.property_type or profile.property_type.strip() not in valid_property_types:
        errors.append(f"Property type must be one of: {', '.join(valid_property_types)}.")

    valid_intents = {"buy", "rent"}
    if not profile.buyOrRent or profile.buyOrRent.strip().lower() not in valid_intents:
        errors.append("buyOrRent must be either 'buy' or 'rent'.")

    return errors

def validate_phone_number(phone: Optional[str]) -> bool:
    if not phone:
        return False
    # Extract only digits
    digits_only = "".join(re.findall(r"\d", phone))
    # Check if the number has 10 to 15 digits
    return 10 <= len(digits_only) <= 15

def apply_defaults_to_profile(profile: UserProfile) -> UserProfile:
    updated_profile = profile.model_copy()

    if not updated_profile.sqft:
        updated_profile.sqft = "2000"

    if not updated_profile.property_type:
        updated_profile.property_type = "Single Family"

    if not updated_profile.buyOrRent:
        updated_profile.buyOrRent = "buy"

    if not updated_profile.bedrooms:
        updated_profile.bedrooms = 3

    if not updated_profile.bathrooms:
        updated_profile.bathrooms = 2

    if not updated_profile.budget:
        if updated_profile.buyOrRent and updated_profile.buyOrRent.lower() == "buy":
            updated_profile.budget = "300000"
        elif updated_profile.buyOrRent and updated_profile.buyOrRent.lower() == "rent":
            updated_profile.budget = "2000"

    return updated_profile

def normalize_user_profile(profile: UserProfile) -> UserProfile:
    return UserProfile(
        name=profile.name,
        phone=str(format_phone(profile.phone)),
        buyOrRent=profile.buyOrRent.strip().lower(),
        property_type=profile.property_type.strip(),
        sqft=str(normalize_sqft(profile.sqft)),
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

# format phone number to e164 format
def format_phone(phone: str, default_country_code: str = "+1") -> str:
    if not phone:
        return ""
    
    # Remove everything except digits
    digits = re.sub(r"[^\d]", "", phone)

    # Prepend country code if missing (assuming 10-digit US number)
    if len(digits) == 10:
        return default_country_code + digits
    elif digits.startswith("1") and len(digits) == 11:
        return "+" + digits
    elif digits.startswith("+") and len(digits) > 1:
        return digits
    else:
        return "+" + digits  # fallback for already full international