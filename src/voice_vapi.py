# Standard library imports
import os
import json

# Third-party library imports
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
import chromadb
import logfire
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage

# Local application imports
from agent.agent_config import AgentDependencies
from models.agent_schedule_config import AgentScheduleConfig
from agent.realtor_agent import realtor_agent
from agent.agent_cost import compute_cost


logfire.configure(send_to_logfire='if-token-present')

load_dotenv()
chroma_db_listings = os.getenv("CHROMA_DB_LISTINGS")
n8n_webhook_url = os.getenv("N8N_WEBHOOK_URL")
agent_timezone = os.getenv("AGENT_TIMEZONE")
port = int(os.getenv("VAPI_EXPOSE_PORT", 8000))


app = FastAPI()

# Session store for multiple callers
session_store: Dict[str, Dict] = {}

class Call(BaseModel):
    id: str
    type: str

class Message(BaseModel):
    role: str
    content: str

class VAPIRequest(BaseModel):
    model: str
    call: Call
    messages: List[Message]
    temperature: float
    max_tokens: int
    metadata: dict
    timestamp: int
    stream: bool

agent_schedule_config = AgentScheduleConfig(
                                timezone=agent_timezone
                        )

agent_dependencies = AgentDependencies(
    chroma_client=chromadb.PersistentClient(path="chroma_db"),
    chroma_db_listings=chroma_db_listings,
    n8n_webhook_url=n8n_webhook_url,
    agent_schedule_config=agent_schedule_config
    )

@app.post("/vapi-webhook/chat/completions")
async def vapi_webhook(req: VAPIRequest):
    
    session_id = str(req.call.id)
    user_message = req.messages[-1].content

    message_history: List[ModelMessage] = []

    # Initialize session if it doesn't exist
    if session_id not in session_store:
        session_store[session_id] = {
            "agent": realtor_agent,
            "usage": Usage(),
            "agent_dependencies": agent_dependencies,
            "message_history": message_history
        }

    session = session_store[session_id]
    agent = session["agent"]
    usage = session["usage"]
    deps = session["agent_dependencies"]
    message_history = session["message_history"]

    # Run agent (non-streaming)
    response = await agent.run(
        user_message,
        deps=deps,
        message_history=message_history,
        usage=usage
    )

    print(f"Caller [{session_id}]: {user_message}")
    print(f"Agent [{session_id}]: {response.output}")

    prompt_cost, completion_cost, total_cost = await compute_cost(usage=usage)
    print(f"prompt_cost: {prompt_cost}, completion_cost: {completion_cost}, total_cost: {total_cost}")
    print(f"request_tokens: {usage.request_tokens}, response_tokens: {usage.response_tokens}, total_tokens: {usage.total_tokens}, requests: {usage.requests}")

    session["message_history"] = response.all_messages()

    final_response = {
        "id": f"chatcmpl-{session_id}",
        "object": "chat.completion.chunk",
        "created": int(req.timestamp / 1000),
        "model": "gpt-4",
        "choices": [
            {
                "delta": {"content": response.output},
                "index": 0,
                "finish_reason": "stop",
            }
        ]
    }

    async def stream():
        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

if __name__ == "__main__":
    uvicorn.run("voice_vapi:app", host="0.0.0.0", port=port, reload=True)
