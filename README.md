# AutoStream Inflx Agent 🎬

> **Social-to-Lead Agentic Workflow** — An AI-powered conversational agent that converts social media conversations into qualified business leads for AutoStream, a fictional SaaS video editing platform.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [How to Run Locally](#how-to-run-locally)
3. [Architecture Explanation](#architecture-explanation)
4. [Conversation Flow Demo](#conversation-flow-demo)
5. [WhatsApp Deployment via Webhooks](#whatsapp-deployment-via-webhooks)
6. [Project Structure](#project-structure)

---

## Project Overview

**Inflx** is AutoStream's AI sales agent that:
- Correctly identifies user intent (greeting / product inquiry / high-intent lead)
- Answers product & pricing questions using a **local RAG pipeline** (no external vector DB needed)
- Detects high-intent users and **collects lead information** across multiple turns
- Triggers a **mock CRM tool** (`mock_lead_capture`) only after all required fields are collected

---

## How to Run Locally

### Prerequisites
- Python 3.9+
- A free [Groq API key](https://console.groq.com) (or OpenAI / Google API key)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/shinde-abhay/autostream-inflx-agent.git
cd autostream-inflx-agent

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env .env
# Edit .env and add your API key

# 5. Run the agent
python main.py
```

### Switching LLM Providers

Edit `.env`:

| Provider | `LLM_PROVIDER` | Required Key    |
|----------|---------------|-----------------|
| Groq (default) | `groq` | `GROQ_API_KEY`  |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Google Gemini | `google` | `GOOGLE_API_KEY` |

---

## Architecture Explanation (~250 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, inspectable state management** via a typed `TypedDict` state object. Every conversation turn carries a complete snapshot — including the full message history, intent label, and partially collected lead fields. This makes multi-turn lead collection deterministic: we always know exactly which fields have been gathered and which are still missing, without relying on the LLM to "remember" them.

### How State is Managed

The `AgentState` TypedDict holds:
- `messages` — the full conversation history (LangChain `BaseMessage` list, append-only)
- `intent` — last classified intent (`greeting | product_inquiry | high_intent_lead | collecting_lead`)
- `lead_name / lead_email / lead_platform` — captured incrementally across turns
- `collecting_lead` / `lead_captured` — boolean flags that drive routing

The graph has **5 nodes**:
```
classify_intent → [router] → respond_greeting
                           → respond_with_rag
                           → collect_lead_info → [sub-router] → capture_lead
```

### RAG Pipeline

The RAG pipeline is a **zero-dependency local retriever**. It loads the knowledge base from `autostream_kb.json` + `autostream_kb.md`, chunks it into ~20 semantic units, builds a TF-IDF index in memory, and retrieves the top-3 most relevant chunks at query time. This means **no OpenAI embeddings, no Pinecone, no setup** — it works offline.

### Tool Calling

`mock_lead_capture()` is **only invoked** after `validate_lead_fields()` confirms all three fields (name, email, platform) are non-null and valid. The `collect_lead_info` node uses the LLM to extract field values from free-text responses and loops until validation passes.

---

## Conversation Flow Demo

```
 You: Hi there!
 Aria: Hey! I'm Aria from AutoStream 👋 How can I help you today?
       [intent: greeting]

 You: What's the difference between the Basic and Pro plans?
 Aria: Great question! Here's a quick breakdown:
       • Basic ($29/mo): 10 videos/month, 720p resolution, email support
       • Pro ($79/mo): Unlimited videos, 4K export, AI captions, 24/7 priority support
       [intent: product_inquiry]

 You: That sounds great! I want to try the Pro plan for my YouTube channel.
 Aria: Awesome, I'd love to get you set up! 🎉 Could I get your full name?
       [intent: high_intent_lead]

 You: Sure, I'm Alex Johnson
 Aria: Thanks Alex! What's your email address?
       [collected: name]

 You: alex@gmail.com
 Aria: Perfect! And which platform are you mainly creating for — YouTube, Instagram, TikTok?
       [collected: name, email]

 You: YouTube, mostly
 Aria: Perfect, I have everything! Let me get you set up now, Alex. 🚀
       [collected: name, email, platform]

══════════════════════════════════════════
  ✅  LEAD CAPTURED SUCCESSFULLY
══════════════════════════════════════════
  Lead ID   : LEAD-04231
  Name      : Alex Johnson
  Email     : alex@gmail.com
  Platform  : YouTube
══════════════════════════════════════════

 Aria: You're all set, Alex! 🎬 Your info is in our system and a team member 
       will reach out within 24 hours. In the meantime, check out 
       docs.autostream.io to explore what's possible. Welcome aboard!

 ✅  Lead capture complete. Session ended.
```

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business Cloud API** (Meta) with a webhook integration:

### Architecture

```
WhatsApp User
     │  (sends message)
     ▼
Meta WhatsApp Business API
     │  POST /webhook  (message payload)
     ▼
FastAPI / Flask Webhook Server  ←─────────┐
     │                                    │
     ├── Verify webhook signature         │
     ├── Extract user phone + message     │
     ├── Look up / create session state   │
     ├── Invoke LangGraph agent.invoke()  │
     ├── Get agent reply                  │
     └── POST reply → WhatsApp API ───────┘
```

### Implementation Steps

1. **Register a webhook** on [Meta for Developers](https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks) — provide your public HTTPS endpoint (e.g., via ngrok for dev, or a VPS for prod).

2. **Create a webhook handler** (FastAPI example):

```python
from fastapi import FastAPI, Request
import httpx, json

app = FastAPI()
SESSIONS = {}   # phone_number → AgentState (use Redis in prod)

@app.post("/webhook")
async def handle_message(request: Request):
    body = await request.json()
    msg   = body["entry"][0]["changes"][0]["value"]["messages"][0]
    phone = msg["from"]
    text  = msg["text"]["body"]

    # Load or create session
    state = SESSIONS.get(phone, initial_state())
    state["messages"].append(HumanMessage(content=text))

    # Run LangGraph agent
    state = graph.invoke(state)
    SESSIONS[phone] = state

    reply = state["last_agent_reply"]

    # Send reply via WhatsApp API
    await send_whatsapp_message(phone, reply)
    return {"status": "ok"}
```

3. **Persist sessions** across webhook calls using Redis (keyed by phone number) — since WhatsApp delivers each message as a separate HTTP request.

4. **Handle media**: For voice messages, use Meta's media download API + a speech-to-text service (Whisper) before passing to the agent.

5. **Deploy** on any HTTPS-capable server (Railway, Render, EC2, GCP Cloud Run).

---

## Project Structure

```
autostream-inflx-agent/
├── main.py                          # CLI entrypoint
├── requirements.txt
├── .env.example
├── README.md
│
├── knowledge_base/
│   ├── autostream_kb.json           # Structured pricing & policy data
│   └── autostream_kb.md             # Markdown knowledge base
│
├── agent/
│   ├── graph.py                     # LangGraph graph definition + all nodes
│   ├── rag_pipeline.py              # Local TF-IDF RAG retriever
│   └── intent_detector.py           # Heuristic + LLM intent classifier
│
└── tools/
    └── lead_capture.py              # mock_lead_capture() + field validation
```

---

## Evaluation Checklist

| Criterion | Implementation |
|-----------|---------------|
| ✅ Intent Detection | Heuristic fast-path + LLM fallback |
| ✅ RAG Knowledge Retrieval | Local TF-IDF, no external DB |
| ✅ State Management | LangGraph `AgentState` TypedDict across 5–6 turns |
| ✅ Tool Calling | `mock_lead_capture` only after all 3 fields validated |
| ✅ Code Structure | Modular package with clear separation of concerns |
| ✅ Real-world Deployability | WhatsApp webhook architecture documented |

---

*Built for ServiceHive's Inflx platform assignment.*
