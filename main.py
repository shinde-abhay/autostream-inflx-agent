"""
main.py
AutoStream Inflx Agent – CLI entrypoint.
"""

import os
import sys
from pathlib import Path

# Step 1: load .env explicitly from project root
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent / ".env"
if not _env_path.exists():
    _env_path = Path.cwd() / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

# Step 2: validate key before importing LLM deps
def _check_env():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    key_map  = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
    key_name = key_map.get(provider, "GROQ_API_KEY")
    val      = os.getenv(key_name, "").strip()
    if not val:
        print(f"\n  Missing {key_name} — not found in {_env_path}")
        print("  Your .env must have NO inline comments. Correct format:\n")
        print("    LLM_PROVIDER=groq")
        print("    GROQ_API_KEY=gsk_...")
        print("    GROQ_MODEL=llama-3.3-70b-versatile\n")
        raise RuntimeError(f"Missing {key_name} in {_env_path}")

_check_env()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage
from agent.graph import build_graph, AgentState

BANNER = """
╔══════════════════════════════════════════════════════════╗
║           AutoStream  •  Inflx AI Agent                  ║
║     Social-to-Lead Conversational Workflow Demo          ║
╚══════════════════════════════════════════════════════════╝
 Type your message and press Enter.  Type 'quit' to exit.
──────────────────────────────────────────────────────────
"""

def run():
    print(BANNER)
    graph = build_graph()
    state: AgentState = {
        "messages": [], "intent": "",
        "lead_name": None, "lead_email": None, "lead_platform": None,
        "lead_captured": False, "collecting_lead": False, "last_agent_reply": ""
    }
    while True:
        try:
            user_input = input("\n You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("\n Aria: Thanks for chatting! Have a great day!\n")
            break
        state["messages"].append(HumanMessage(content=user_input))
        try:
            state = graph.invoke(state)
        except Exception as e:
            import traceback
            print(f"\n [Error] {e}")
            traceback.print_exc()
            continue
        reply = state.get("last_agent_reply", "")
        if reply:
            print(f"\n Aria: {reply}")
        intent = state.get("intent", "")
        if intent:
            print(f"       [intent: {intent}]", end="")
        if state.get("collecting_lead") and not state.get("lead_captured"):
            collected = [f for f in ("lead_name","lead_email","lead_platform") if state.get(f)]
            if collected:
                print(f"  [collected: {', '.join(c.replace('lead_','') for c in collected)}]", end="")
        print()
        if state.get("lead_captured"):
            print("\n  Lead capture complete. Session ended.\n")
            break

if __name__ == "__main__":
    run()
