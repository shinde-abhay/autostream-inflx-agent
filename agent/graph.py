"""
agent/graph.py
LangGraph-based agentic workflow for AutoStream's Inflx social-to-lead system.

Graph nodes:
  classify_intent → route →
    ├─ respond_greeting
    ├─ respond_with_rag
    ├─ collect_lead_info  (multi-turn, loops until all fields collected)
    └─ capture_lead       (calls mock_lead_capture tool)
"""

from __future__ import annotations

import os
import json
from typing import TypedDict, Annotated, Optional, List
import operator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from langgraph.graph import StateGraph, END

from agent.rag_pipeline  import query_kb, init_rag
from agent.intent_detector import classify_intent_heuristic
from tools.lead_capture  import mock_lead_capture, validate_lead_fields


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:          Annotated[List[BaseMessage], operator.add]   # full conversation
    intent:            str                                           # current intent label
    lead_name:         Optional[str]
    lead_email:        Optional[str]
    lead_platform:     Optional[str]
    lead_captured:     bool
    collecting_lead:   bool                                          # True while in lead collection loop
    last_agent_reply:  str                                           # for display


# ─── LLM setup ────────────────────────────────────────────────────────────────

def get_llm():
    """Return the configured LLM. Supports Groq (Claude/Llama), OpenAI, or Google."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.4,
            api_key=os.getenv("GROQ_API_KEY")
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.4,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


# ─── Shared system persona ────────────────────────────────────────────────────

AGENT_PERSONA = """You are Aria, the friendly AI sales assistant for AutoStream — a SaaS platform 
that automates video editing for content creators. You are warm, concise, and helpful.
Your goal is to answer questions accurately and, when users show buying intent, 
smoothly guide them through signing up. Never make up information not in the knowledge base."""


# ─── Node: Classify Intent ────────────────────────────────────────────────────

def classify_intent_node(state: AgentState) -> AgentState:
    """Determine the intent of the latest user message."""
    last_msg = state["messages"][-1].content if state["messages"] else ""

    # Fast heuristic first
    heuristic = classify_intent_heuristic(last_msg)

    # If collecting lead info, stay in that mode regardless
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return {**state, "intent": "collecting_lead"}

    # For ambiguous heuristic results, use LLM
    if heuristic == "other":
        llm    = get_llm()
        prompt = (
            "Classify this user message for AutoStream (video editing SaaS) into one of: "
            "greeting, product_inquiry, high_intent_lead, or other.\n"
            f"Message: {last_msg}\n"
            "Reply with ONLY the label."
        )
        result = llm.invoke([HumanMessage(content=prompt)])
        intent = result.content.strip().lower().split()[0]
        # Sanitize
        if intent not in {"greeting", "product_inquiry", "high_intent_lead"}:
            intent = "other"
    else:
        intent = heuristic

    return {**state, "intent": intent}


# ─── Router ───────────────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    """Conditional edge: map intent → next node name."""
    if state.get("lead_captured"):
        return END
    if state["intent"] == "collecting_lead" or state.get("collecting_lead"):
        return "collect_lead_info"
    if state["intent"] == "greeting":
        return "respond_greeting"
    if state["intent"] in ("product_inquiry", "other"):
        return "respond_with_rag"
    if state["intent"] == "high_intent_lead":
        return "collect_lead_info"
    return "respond_with_rag"


# ─── Node: Greeting ───────────────────────────────────────────────────────────

def respond_greeting(state: AgentState) -> AgentState:
    llm    = get_llm()
    system = SystemMessage(content=AGENT_PERSONA)
    prompt = HumanMessage(content=(
        "The user just greeted you. Respond warmly, introduce yourself briefly as Aria from AutoStream, "
        "and ask how you can help them today. Keep it to 2-3 sentences."
    ))
    result = llm.invoke([system] + state["messages"] + [prompt])
    reply  = result.content.strip()
    return {
        **state,
        "messages":         state["messages"] + [AIMessage(content=reply)],
        "last_agent_reply": reply
    }


# ─── Node: RAG-powered response ───────────────────────────────────────────────

def respond_with_rag(state: AgentState) -> AgentState:
    llm       = get_llm()
    user_msg  = state["messages"][-1].content
    kb_context = query_kb(user_msg, top_k=4)

    system = SystemMessage(content=(
        f"{AGENT_PERSONA}\n\n"
        "Use ONLY the following knowledge base context to answer. "
        "If the answer isn't in the context, say so honestly.\n\n"
        f"KNOWLEDGE BASE CONTEXT:\n{kb_context}"
    ))

    result = llm.invoke([system] + state["messages"])
    reply  = result.content.strip()

    return {
        **state,
        "messages":         state["messages"] + [AIMessage(content=reply)],
        "last_agent_reply": reply
    }


# ─── Node: Collect Lead Info ──────────────────────────────────────────────────

def collect_lead_info(state: AgentState) -> AgentState:
    """
    Multi-turn node that asks for name, email, and platform one-by-one.
    Parses LLM responses to extract field values. Loops until all are collected,
    then hands off to capture_lead.
    """
    llm      = get_llm()
    messages = state["messages"]
    user_msg = messages[-1].content if messages else ""

    name     = state.get("lead_name")
    email    = state.get("lead_email")
    platform = state.get("lead_platform")

    # ── Try to extract values from the user's message ──
    extraction_prompt = (
        "Extract any of the following from the user message (return JSON, null if not present):\n"
        '{"name": "...", "email": "...", "platform": "..."}\n\n'
        f'User message: "{user_msg}"\n\n'
        "Rules:\n"
        "- name: First + last name or just first name\n"
        "- email: valid email address\n"
        "- platform: YouTube, Instagram, TikTok, Twitter, LinkedIn, or similar\n"
        "Reply ONLY with the JSON object."
    )
    extraction = llm.invoke([HumanMessage(content=extraction_prompt)])
    try:
        raw = extraction.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(raw)
        if extracted.get("name")     and not name:
            name = extracted["name"]
        if extracted.get("email")    and "@" in extracted["email"] and not email:
            email = extracted["email"]
        if extracted.get("platform") and not platform:
            platform = extracted["platform"]
    except Exception:
        pass  # Extraction failed; we'll just ask again

    # ── Determine what still needs to be collected ──
    validation = validate_lead_fields(name, email, platform)

    if validation["valid"]:
        # All fields collected — hand off to capture node
        reply = (
            f"Perfect, I have everything I need! Let me get you set up now, {name}. "
            "Completing your registration... 🚀"
        )
        return {
            **state,
            "lead_name":        name,
            "lead_email":       email,
            "lead_platform":    platform,
            "collecting_lead":  True,
            "messages":         messages + [AIMessage(content=reply)],
            "last_agent_reply": reply,
            "intent":           "capture_lead"
        }

    # ── Build a natural prompt for the next missing field ──
    missing = validation["missing"]
    if "name" in " ".join(missing):
        ask_for = "your full name"
    elif "email" in " ".join(missing):
        ask_for = "your email address"
    else:
        ask_for = "which creator platform you primarily use (e.g., YouTube, Instagram, TikTok)"

    system = SystemMessage(content=(
        f"{AGENT_PERSONA}\n\n"
        f"You are collecting signup information. You still need: {', '.join(missing)}.\n"
        f"Already collected: name={name}, email={email}, platform={platform}.\n"
        f"Ask specifically for {ask_for} in a natural, friendly way. Keep it to one question."
    ))

    result = llm.invoke([system] + messages)
    reply  = result.content.strip()

    return {
        **state,
        "lead_name":        name,
        "lead_email":       email,
        "lead_platform":    platform,
        "collecting_lead":  True,
        "messages":         messages + [AIMessage(content=reply)],
        "last_agent_reply": reply
    }


# ─── Node: Capture Lead ───────────────────────────────────────────────────────

def capture_lead(state: AgentState) -> AgentState:
    """Calls the mock CRM tool and generates a confirmation message."""
    result = mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"]
    )

    llm    = get_llm()
    system = SystemMessage(content=AGENT_PERSONA)
    prompt = HumanMessage(content=(
        f"The lead has been successfully captured in our CRM (Lead ID: {result['lead_id']}).\n"
        f"User: {state['lead_name']}, Email: {state['lead_email']}, Platform: {state['lead_platform']}.\n"
        "Generate a warm, enthusiastic confirmation message. Tell them:\n"
        "1. Their info has been received\n"
        "2. A team member will reach out within 24 hours\n"
        "3. They can also check out docs.autostream.io in the meantime\n"
        "Keep it to 3-4 sentences."
    ))
    reply = llm.invoke([system, prompt]).content.strip()

    return {
        **state,
        "lead_captured":    True,
        "collecting_lead":  False,
        "messages":         state["messages"] + [AIMessage(content=reply)],
        "last_agent_reply": reply
    }


# ─── Capture-lead router ──────────────────────────────────────────────────────

def route_collect(state: AgentState) -> str:
    """After collect_lead_info: go to capture if intent flipped, else loop."""
    if state.get("intent") == "capture_lead":
        return "capture_lead"
    return END  # return to user to get next input


# ─── Build the graph ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    init_rag()

    builder = StateGraph(AgentState)

    builder.add_node("classify_intent",  classify_intent_node)
    builder.add_node("respond_greeting", respond_greeting)
    builder.add_node("respond_with_rag", respond_with_rag)
    builder.add_node("collect_lead_info", collect_lead_info)
    builder.add_node("capture_lead",     capture_lead)

    builder.set_entry_point("classify_intent")

    builder.add_conditional_edges("classify_intent", route, {
        "respond_greeting":  "respond_greeting",
        "respond_with_rag":  "respond_with_rag",
        "collect_lead_info": "collect_lead_info",
        END:                 END
    })

    builder.add_edge("respond_greeting", END)
    builder.add_edge("respond_with_rag", END)

    builder.add_conditional_edges("collect_lead_info", route_collect, {
        "capture_lead": "capture_lead",
        END:            END
    })

    builder.add_edge("capture_lead", END)

    return builder.compile()
