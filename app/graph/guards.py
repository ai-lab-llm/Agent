import re, ast
from typing import Tuple

SQL_HEAD   = re.compile(r"(?is)^\s*(select|with|pragma|explain)\b")
ANSWER_SQL = re.compile(r"(?is)^\s*answer:\s*(select|with|pragma|explain)\b")
STRIP_TAG  = re.compile(r"^\s*(System:|Human:|AI:|Tool:)\s*", re.I)

ALLOWED_SCHEMA = {
    "users": {"columns": ["id", "name"], "pk": ["id"]},
    "event": {
        "columns": [
            "id","protectee_id","timestamp","ppg_json","ppg_threat_detected",
            "hrv","stress","imu_danger_level","latitude","longitude",
            "zone_type","is_watch_connected"
        ],
        "pk": ["id"],
    },
}
_ALLOWED_TABLES = set(ALLOWED_SCHEMA.keys())
_ALLOWED_COLS = {t: set(ALLOWED_SCHEMA[t]["columns"]) for t in _ALLOWED_TABLES}
_DOT_COL = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")

DDL_DML = re.compile(r"(?i)\b(create|insert|update|delete|drop|alter|truncate|attach|detach)\b")

from app.graph.schema_facts import JOIN_RULE


def validate_sql_against_schema(sql: str) -> Tuple[bool, str]:
    if DDL_DML.search(sql):
        return False, "DDL/DML not allowed"
    unknown = []
    for tbl, col in _DOT_COL.findall(sql):
        if tbl in _ALLOWED_TABLES:
            if col not in _ALLOWED_COLS[tbl]:
                unknown.append(f"{tbl}.{col}")
        else:
            if not any(col in _ALLOWED_COLS[t] for t in _ALLOWED_TABLES):
                unknown.append(f"{tbl}.{col} (alias?)")
    if unknown:
        return False, "Unknown columns/tables: " + ", ".join(sorted(set(unknown)))
    return True, ""


def extract_sql(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"^```.*?```", "", s, flags=re.S | re.M)
    cleaned = []
    for ln in s.splitlines():
        if STRIP_TAG.match(ln):
            continue
        cleaned.append(ln.strip())
    start = next((i for i, ln in enumerate(cleaned) if SQL_HEAD.match(ln)), None)
    if start is None:
        return ""
    sql = "\n".join(cleaned[start:]).strip()
    sql = re.sub(r"(?is)^\s*answer:\s*", "", sql)
    sql = re.sub(r"(?is)^\s*query:\s*", "", sql)
    return sql


def parse_tool_result(text: str):
    text = str(text).strip()
    if text.lower().startswith("error:"):
        return False, text
    try:
        rows = ast.literal_eval(text)
        return True, rows
    except Exception:
        return True, text 