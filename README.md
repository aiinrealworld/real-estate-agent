
# 🏡 Real Estate AI Voice & Chat Agent

This project is an AI-powered **Real Estate Agent** capable of handling buyer inquiries via **text chat** and **voice (VAPI)**. It uses natural language understanding, vector search for property recommendations, and integrates with a calendar system via **n8n** to schedule property showings.

---

## ✨ Features

- 🔍 **Understands buyer preferences** from natural conversations (budget, location, property type, etc.)
- 🧠 **Recommends properties** using vector embeddings and ChromaDB
- 📅 **Checks agent availability** and schedules appointments with buffer logic
- 📞 **Voice-ready** via [VAPI](https://vapi.ai/)
- 🧪 **Interactive test chat CLI** via terminal
- 🔄 Integrated with **n8n** for real-time calendar access and scheduling

---

## 🗂️ Project Structure

```
real_estate_agent/
├── agent/
│   ├── realtor_agent.py                 # Pydantic AI agent definition
│   └── tools/                           # Agent tools (LLM-callable functions)
│       ├── recommend_properties.py
│       ├── get_agent_availability.py
│       └── schedule_appointment.py
│
├── models/                              # Pydantic models
│   ├── user_profile.py
│   ├── property_recommendation.py
│   └── agent_schedule_config.py
│
├── utils/                               # Support logic (slots, embeddings, etc.)
│   ├── embedding_utils.py
│   ├── time_utils.py
│   └── appointment_utils.py
│
├── agent_config.py                      # System prompt, agent dependencies
├── chat.py                              # CLI-based text testing for the agent
├── voice_vapi.py                        # FastAPI server to handle VAPI voice requests
└── .env                                 # Environment variable configuration (not checked in)
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/real-estate-agent.git
cd real-estate-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file in the root with the following:

```env
OPENAI_API_KEY=your-openai-key
OPEN_ROUTER_LLM_MODEL=gpt-3.5-turbo  # or your chosen model
CHROMA_DB_LISTINGS=real_estate_listings
N8N_WEBHOOK_URL=https://your-n8n-instance/webhook/calendar
AGENT_TIMEZONE=America/Chicago
```

---

## 💬 Run Chat (Text Interface)

```bash
python chat.py
```

This launches a terminal-based chatbot you can interact with using natural language.

---

## 📞 Run Voice Agent (VAPI Webhook)

```bash
python voice_vapi.py
```

- Exposes a FastAPI webhook on `http://localhost:8000/vapi-webhook/chat/completions`
- Configure VAPI to send POST requests to this endpoint during live calls

---

## 📦 Core Technologies

| Layer | Tool |
|-------|------|
| **LLM Agent Framework** | [Pydantic AI](https://github.com/ericmjl/pydantic-ai) |
| **LLM Provider** | OpenAI (via OpenRouter or directly) |
| **Embedding Search** | [ChromaDB](https://docs.trychroma.com/) |
| **Voice Integration** | [VAPI](https://vapi.ai/) |
| **Scheduling Backend** | [n8n](https://n8n.io/) |
| **Environment Handling** | `dotenv` |
| **Web Framework** | FastAPI |
| **Observability** | `logfire` |

---

## 🧠 Agent Architecture

The agent uses **tool-augmented prompting**. Based on user input, it can autonomously call tools like:

1. `recommend_properties`: Finds top listings using vector search
2. `get_agent_availability`: Computes availability with buffer and calendar integration
3. `schedule_appointment`: Books a showing via n8n calendar webhook

The agent is defined in `agent/realtor_agent.py` and loaded in both `chat.py` and `voice_vapi.py`.

---

## ✅ Sample Conversation

**You**: I’m looking for a 2-bedroom condo in Chicago under 500K  
**Agent**: Great! Let me find some listings for you…  
**Agent**: Here are a few options. Would you like to book a tour for one of them?

---

## 🧪 Testing & Debugging

- Use `chat.py` to test logic quickly without voice
- Logs (via `logfire`) can be configured using `LOGFIRE_PROJECT_ID` and token in your `.env`
- Agent tools can be tested independently in isolation

---

## 🛠 Future Enhancements (Ideas)

- Real-time listing updates via MLS or IDX API
- Persistent user profiles / conversation memory
- Support for renters or sellers
- Frontend dashboard or mobile interface
- Multi-agent delegation (e.g., one agent recommends, another books)

---

## 📄 License

MIT License — free to use and modify for your own real estate or voice automation projects.