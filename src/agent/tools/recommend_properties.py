# Third-party library imports
from pydantic_ai import RunContext

# Local application imports
from agent.agent_config import AgentDependencies
from utils.embedding_utils import get_embedding, profile_to_text
from agent.realtor_agent import realtor_agent
from models.property_recommendation import parse_chroma_results
from models.user_profile import (
    UserProfile,
    apply_defaults_to_profile,
    normalize_user_profile,
    validate_user_profile,
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