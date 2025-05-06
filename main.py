import asyncio
from agent import realtor_agent
from models import UserProfile
from dotenv import load_dotenv
import logfire
from typing import List
from pydantic_ai.messages import ModelMessage

# Optionally configure logfire
logfire.configure(send_to_logfire='if-token-present')


async def main():
    # Initialize agent and user profile
    agent = realtor_agent
    user_profile = UserProfile()

    profile_fields = [
        "name", "phone", "location", "property_type", "budget",
        "bedrooms", "bathrooms", "must_haves", "good_to_haves"
    ]

    print("Welcome to the Real Estate Agent Chat!")
    message = ""

    message_history: List[ModelMessage] = []

    # Chat loop
    while True:

        # Create a summary of the current UserProfile
        profile_summary = f"""
        Current User Profile:
        - Name: {user_profile.name}
        - Phone: {user_profile.phone}
        - Location: {user_profile.location}
        - Property Type: {user_profile.property_type}
        - Budget: {user_profile.budget}
        - Bedrooms: {user_profile.bedrooms}
        - Bathrooms: {user_profile.bathrooms}
        - Must-haves: {user_profile.must_haves}
        - Good-to-haves: {user_profile.good_to_haves}
        """

        # Augment the message with the profile summary
        augmented_message = message + "\n" + profile_summary

        response = await agent.run(
            augmented_message, 
            deps=user_profile,
            message_history=message_history)
        
        message_history = response.all_messages()
        
        print(f"Agent: {response.output}")

        # Debug: Show profile state after each run
        print(f"[DEBUG] Current Profile: {user_profile}\n")

        # Check if all required fields are filled
        basic_fields_filled = all(
            getattr(user_profile, field) not in (None, "") for field in profile_fields[:7]
        )
        list_fields_filled = all(
            len(getattr(user_profile, field)) > 0 for field in profile_fields[7:]
        )

        # Prompt next input
        message = input("You: ")


if __name__ == "__main__":
    asyncio.run(main())
