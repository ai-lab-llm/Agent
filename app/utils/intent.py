from __future__ import annotations
from typing import Dict, Any
import json, re
from app.core.llm import get_chat_llm
from app.core.database import get_db
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

SCHEMA_HINT = """
You are a gatekeeper for a database Q&A endpoint.

This endpoint ONLY accepts questions that can be answered by querying a small SQLite DB
with tables: users(id, name), event(protectee_id -> users.id, timestamp, ppg_json, ppg_threat_detected, hrv, stress, imu_danger_level, latitude, longitude, zone_type, is_watch_connected).

Label the user's input as either:
- "db_query": if the intent is to retrieve/aggregate/filter something from these tables/columns (highest/lowest/average, latest time, count, list within date range, filter by user name, zone_type, watch connection, etc.)
- "other": greetings, chit-chat, general questions, non-database tasks, or requests without any retrievable target from this schema.

ALWAYS return pure JSON: {{\"intent\": \"db_query\"}} or {{\"intent\": \"other\"}}
Do not add explanations. No code fences.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SCHEMA_HINT),
    ("system",
     "Examples:\n"
     "Q: \"박해름의 스트레스가 가장 높았던 시각 알려줘\" -> {{\"intent\":\"db_query\"}}\n"
     "Q: \"안녕?\" -> {{\"intent\":\"other\"}}\n"
     "Q: \"요즘 날씨 어때?\" -> {{\"intent\":\"other\"}}\n"
     "Q: \"박주연의 HRV 최저값과 시각\" -> {{\"intent\":\"db_query\"}}\n"
     "Q: \"워치가 최근에 끊긴 시간\" -> {{\"intent\":\"db_query\"}}\n"
     "Q: \"수학 문제 풀어줘\" -> {{\"intent\":\"other\"}}\n"
    ),
    ("human", "{question}")
])

def _robust_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(0)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and data.get("intent") in {"db_query", "other"}:
            return data
    except Exception:
        pass
    return {"intent": "other"}

def classify_intent_llm(question: str) -> str:
    """Return 'db_query' or 'other'."""
    llm = get_chat_llm()
    chain = PROMPT | llm | StrOutputParser() | RunnableLambda(_robust_json)
    out = chain.invoke({"question": question})
    return out.get("intent", "other")

def list_known_names(limit: int = 5) -> list[str]:
    names: list[str] = []
    try:
        rows = get_db().run("SELECT name FROM users LIMIT 20")
        for r in rows or []:
            if isinstance(r, (list, tuple)) and r and isinstance(r[0], str):
                names.append(r[0])
    except Exception:
        pass
    return names[:limit]



_MAX_PAT = re.compile(r"(가장\s*높|최대|최고|highest|max)", re.I)
_MIN_PAT = re.compile(r"(가장\s*낮|최소|lowest|min)", re.I)
_WHEN_PAT = re.compile(r"(언제|시각|시간|몇\s*시|시점|때)", re.I)

def detect_extreme_direction(question: str) -> str | None:
    if _MAX_PAT.search(question):
        return "max"
    if _MIN_PAT.search(question):
        return "min"
    return None

def asks_when(question: str) -> bool:
    return _WHEN_PAT.search(question) is not None
