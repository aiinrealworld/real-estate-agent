import asyncio
from agent import realtor_agent
from models.user_profile import UserProfile
from dotenv import load_dotenv
import logfire
from typing import List
from pydantic_ai.messages import ModelMessage

logfire.configure(send_to_logfire='if-token-present')

async def main():
    # Initialize agent and user profile
    agent = realtor_agent
    user_profile = UserProfile()

    print("Welcome to the Real Estate Agent Chat!")
    message = ""

    message_history: List[ModelMessage] = []

    # Chat loop
    while True:

        response = await agent.run(
            message, 
            deps=user_profile,
            message_history=message_history)
        
        message_history = response.all_messages()
        
        print(f"Agent: {response.output}")

        # Prompt next input
        message = input("You: ")


if __name__ == "__main__":
    asyncio.run(main())
