"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
In production, this would POST to a CRM API (HubSpot, Salesforce, etc.)
"""

import json
import datetime
from typing import Optional


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a lead in a CRM system.
    
    Args:
        name: Full name of the lead
        email: Email address of the lead
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)
    
    Returns:
        dict with status and lead_id
    """
    timestamp = datetime.datetime.now().isoformat()
    lead_id = f"LEAD-{abs(hash(email)) % 100000:05d}"

    # Simulate CRM write
    lead_data = {
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
        "source": "inflx_agent",
        "status": "new",
        "plan_interest": "Pro"
    }

    # Pretty print to console (simulates CRM log)
    print("\n" + "=" * 55)
    print("  ✅  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Lead ID   : {lead_id}")
    print(f"  Name      : {name}")
    print(f"  Email     : {email}")
    print(f"  Platform  : {platform}")
    print(f"  Timestamp : {timestamp}")
    print("=" * 55 + "\n")

    return {
        "success": True,
        "lead_id": lead_id,
        "message": f"Lead captured successfully: {name}, {email}, {platform}"
    }


def validate_lead_fields(name: Optional[str], email: Optional[str], platform: Optional[str]) -> dict:
    """
    Validates that all required lead fields are present and well-formed.
    Returns a dict with 'valid' bool and 'missing' list.
    """
    missing = []
    if not name or name.strip() == "":
        missing.append("name")
    if not email or "@" not in email:
        missing.append("email (valid address)")
    if not platform or platform.strip() == "":
        missing.append("creator platform (e.g., YouTube, Instagram)")
    
    return {
        "valid": len(missing) == 0,
        "missing": missing
    }
