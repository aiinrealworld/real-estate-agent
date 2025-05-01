from typing import List, Optional
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    must_haves: List[str] = []
    good_to_haves: List[str] = []