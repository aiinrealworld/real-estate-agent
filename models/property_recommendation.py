from pydantic import BaseModel
from typing import List

class PropertyRecommendation(BaseModel):
    listing_id: str
    address: str
    neighborhood: str
    city: str
    state: str
    zip_code: str
    price: int
    bedrooms: int
    bathrooms: float
    square_feet: int
    lot_size: float
    year_built: int
    property_type: str
    mls_status: str
    days_on_market: int
    latitude: float
    longitude: float
    description: str

def parse_chroma_results(chroma_results: dict) -> List[PropertyRecommendation]:
    documents = chroma_results["documents"][0]
    metadatas = chroma_results["metadatas"][0]

    recommendations = []
    for doc, meta in zip(documents, metadatas):
        recommendation = PropertyRecommendation(
            listing_id=meta["listing_id"],
            address=meta["address"],
            city=meta["city"],
            state=meta["state"],
            zip_code=meta["zip_code"],
            neighborhood=meta["neighborhood"],
            property_type=meta["property_type"],
            bedrooms=meta["bedrooms"],
            bathrooms=meta["bathrooms"],
            square_feet=meta["square_feet"],
            lot_size=meta["lot_size"],
            price=meta["price"],
            year_built=meta["year_built"],
            mls_status=meta["mls_status"],
            days_on_market=meta["days_on_market"],
            latitude=meta["latitude"],
            longitude=meta["longitude"],
            description=meta["description"],
            full_text=doc,
        )
        recommendations.append(recommendation)

    return recommendations
