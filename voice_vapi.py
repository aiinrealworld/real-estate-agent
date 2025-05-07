from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from typing import Dict
from typing import List
import uvicorn
from pydantic_ai.messages import ModelMessage

from agent import realtor_agent
from models import UserProfile
import logfire

logfire.configure(send_to_logfire='if-token-present')

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

@app.post("/vapi-webhook/chat/completions")
async def vapi_webhook(req: VAPIRequest):
    
    session_id = str(req.call.id)
    user_message = req.messages[-1].content

    message_history: List[ModelMessage] = []

    # Initialize session if it doesn't exist
    if session_id not in session_store:
        session_store[session_id] = {
            "agent": realtor_agent,
            "profile": UserProfile(),
            "message_history": message_history
        }

    session = session_store[session_id]
    agent = session["agent"]
    user_profile = session["profile"]
    message_history = session["message_history"]

    # Run agent (non-streaming)
    response = await agent.run(
        user_message,
        deps=user_profile,
        message_history=message_history
    )

    print(f"Caller [{session_id}]: {user_message}")
    print(f"Agent [{session_id}]: {response.output}")
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
    uvicorn.run("voice_webhook:app", host="0.0.0.0", port=8000, reload=True)
