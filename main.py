import asyncio
from agent import RealEstateAgent
from agent import UserProfile
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")

async def main():
    agent = RealEstateAgent(model=llm_model) 
    print("Available tools:", list(agent._function_tools))

    user_profile = UserProfile()
    profile_fields = ["name", "phone", "location", "property_type", "bedrooms", "bathrooms", "must_haves", "good_to_haves"]

    print("Welcome to the Real Estate Agent Chat!")
    message = "Hello! I'm here to help you find your dream home. Let's start by gathering some information. What is your name?" # Initial message

    while True:
        response = await agent.run(message, deps=user_profile) # Pass the user_profile as deps
        print(response._output_tool_name) 
        print(f"Agent: {response}")

        # Check if all profile fields are filled.
        all_fields_filled = all(getattr(user_profile, field) is not None and getattr(user_profile, field) != []  for field in profile_fields[:6]) and len(user_profile.must_haves) > 0 and len(user_profile.good_to_haves) > 0

        if all_fields_filled:
            print("Great! I have all the information I need.")
            print(f"Final User Profile: {user_profile}")  # Print the filled UserProfile
            break

        message = input("User: ")


if __name__ == "__main__":
    asyncio.run(main())