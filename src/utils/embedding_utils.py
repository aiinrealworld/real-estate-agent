# Third-party library imports
import openai

# Local application imports
from models.user_profile import UserProfile


# Function to embed using OpenAI
def get_embedding(text: str):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


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