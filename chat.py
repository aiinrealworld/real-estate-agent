# Standard library imports
import asyncio
import os

# Third-party library imports
from dotenv import load_dotenv
import chromadb
import logfire
from typing import List
from pydantic_ai.messages import ModelMessage

# Local application imports
from agent.realtor_agent import realtor_agent
from agent_config import AgentDependencies
from models.agent_schedule_config import AgentScheduleConfig

logfire.configure(send_to_logfire='if-token-present')

async def main():

    load_dotenv()
    chroma_db_listings = os.getenv("CHROMA_DB_LISTINGS")
    n8n_webhook_url = os.getenv("N8N_WEBHOOK_URL")
    agent_timezone = os.getenv("AGENT_TIMEZONE")
    agent_schedule_config = AgentScheduleConfig(
        timezone=agent_timezone
    ) 

    # Initialize agent and user profile
    agent = realtor_agent
    agent_deps = AgentDependencies(
        chroma_client=chromadb.PersistentClient(path="chroma_db"),
        chroma_db_listings=chroma_db_listings,
        n8n_webhook_url=n8n_webhook_url,
        agent_schedule_config=agent_schedule_config
    )

    print("Welcome to the Real Estate Agent Chat!")
    message = "Hello"

    message_history: List[ModelMessage] = []

    # Chat loop
    while True:

        if message.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = await agent.run(
            message, 
            deps=agent_deps,
            message_history=message_history)
        
        message_history = response.all_messages()
        
        print(f"Agent: {response.output}")

        # Prompt next input
        message = input("You: ")


if __name__ == "__main__":
    asyncio.run(main())
