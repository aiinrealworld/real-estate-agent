# Standard library imports
import asyncio
import os

# Third-party library imports
from dotenv import load_dotenv
import chromadb
import logfire
from typing import List
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage

# Local application imports
from agent.realtor_agent import realtor_agent
from agent.agent_config import AgentDependencies
from models.agent_schedule_config import AgentScheduleConfig
from agent.agent_cost import compute_cost

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
    agent_usage = Usage()

    # Chat loop
    while True:

        if message.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = await agent.run(
            message, 
            deps=agent_deps,
            message_history=message_history,
            usage=agent_usage
        )
        
        message_history = response.all_messages()
        
        print(f"Agent: {response.output}")

        # Prompt next input
        message = input("You: ")

    prompt_cost, completion_cost, total_cost = await compute_cost(usage=agent_usage)
    print(f"prompt_cost: {prompt_cost}, completion_cost: {completion_cost}, total_cost: {total_cost}")
    print(f"request_tokens: {agent_usage.request_tokens}, response_tokens: {agent_usage.response_tokens}, total_tokens: {agent_usage.total_tokens}, requests: {agent_usage.requests}")
    


if __name__ == "__main__":
    asyncio.run(main())
