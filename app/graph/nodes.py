# app/graph/nodes.py

import uuid, re
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from app.core.llm import get_chat_llm
from app.core.database import get_db
from app.core.tools import db_query_tool
from app.graph.schema_facts import inject_schema_facts, JOIN_RULE
from app.graph.guards import (
    SQL_HEAD, ANSWER_SQL, STRIP_TAG, validate_sql_against_schema, extract_sql
)
from app.graph.routing import choose_metric
from app.utils.dates import (
    extract_date_yyyy_mm_dd,
    resolve_week_window_kst,
    to_yyyy_mm_dd_hh_mm_ss_strings,
    extract_time_filter,
)

# ----------------------------
# Helpers: intent detection
# ----------------------------
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


def get_sql_tools():
    db = get_db()
    llm = get_chat_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
    get_schema_tool  = next(t for t in tools if t.name == "sql_db_schema")
    return list_tables_tool, get_schema_tool


def handle_tool_error(state) -> dict:
    err = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Here is the error: {repr(err)}\n\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


# === Nodes ===

def first_tool_call(state) -> dict[str, List[AIMessage]]:
    return {
        "messages": [AIMessage(content="", tool_calls=[{
            "name": "sql_db_list_tables", "args": {}, "id": "initial_tool_call_abc123"
        }])]
    }


# schema selection prompt
schema_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at choosing relevant tables. Given a user question and a list of available tables, decide which tables are relevant. Exclude internal SQLite tables like 'sqlite_sequence'. Return only a comma-separated list of table names with NO extra words."),
    ("human", "Question: {question}\nAvailable tables: {tables}")
])



def model_get_schema(state):
    llm = get_chat_llm()
    # latest question
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    # list_tables_tool result
    tables_raw = state["messages"][-1].content
    tables = [t.strip() for t in tables_raw.split(",") if t.strip()]
    tables = [t for t in tables if t.lower() != "sqlite_sequence"]

    need_event = any(k in question for k in ["stress","스트레스","시간","timestamp","hrv","ppg","움직임","위험","흔들림","넘어짐"])
    if need_event and ("event" in tables and "users" in tables):
        selected_str = "event, users"
    else:
        selected = (schema_prompt | llm | StrOutputParser()).invoke({"question": question, "tables": ", ".join(tables)})
        raw_list = [t.strip() for t in selected.split(",") if t.strip()]
        dedup = []
        for t in raw_list:
            if t in tables and t not in dedup:
                dedup.append(t)
        final = dedup if dedup else tables
        selected_str = ", ".join(final)

    # call schema tool
    list_tables_tool, get_schema_tool = get_sql_tools()
    schema_tool_name = getattr(get_schema_tool, "name", "sql_db_schema")
    return {
        "messages": [AIMessage(content="", tool_calls=[{
            "name": schema_tool_name, "args": {"table_names": selected_str},
            "id": f"get_schema_{uuid.uuid4()}",
        }])]
    }


# ======================
# Query generation
# ======================

QUERY_GEN_INSTRUCTION = """You are a SQL expert.

YOU MUST follow these constraints strictly:
- Use ONLY tables/columns explicitly listed in the SCHEMA (STRICT) message below.
- NEVER invent tables or columns. If something is missing, output: Error: Missing data
- Always fully-qualify columns with aliases: event AS e, users AS u.
- To filter by a user name, JOIN users u ON u.id = e.protectee_id and filter u.name = '<name>'.
- e.timestamp is a TEXT datetime ('YYYY-MM-DD HH:MM:SS').

Decide the SQL SHAPE from the user's wording (YOU choose the right form):
- "평균/average" → use AVG(e.{metric_col})
- "가장 높/최대/최고" → ORDER BY e.{metric_col} DESC, then e.timestamp DESC, LIMIT 1
- "가장 낮/최소" → ORDER BY e.{metric_col} ASC, then e.timestamp DESC, LIMIT 1
- "개수/횟수" → COUNT(*)
- "최근/가장 최근/마지막 시각" → ORDER BY e.timestamp DESC, LIMIT 1
- If the user asks explicitly for the time/when ("시간/시각/언제"), include e.timestamp in the SELECT; otherwise only select what is necessary for the answer.
- If the query ranks by the TARGET METRIC (e.g., highest/lowest) AND the user asks "when/언제/날짜/시각", SELECT **both** e.timestamp AND e.{metric_col}.
- NEVER select non-aggregated columns together with aggregates unless you also provide a proper GROUP BY. Prefer removing non-aggregated columns when not needed.

If a resolved week/day time window is provided, you MUST add BOTH filters:
- AND e.timestamp >= '{resolved_from}' (if provided and not empty)
- AND e.timestamp <  '{resolved_to}'   (if provided and not empty)

If a resolved time-of-day filter is provided, compare using strftime:
- AND strftime('%H:%M:%S', e.timestamp) {time_op} '{time_hhmmss}'

- Exclude NULL or blank timestamps: add "AND e.timestamp IS NOT NULL AND e.timestamp <> ''".
- Prefer returning a single, valid SQLite SELECT (no backticks, no commentary). No DDL/DML statements.
"""

query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", QUERY_GEN_INSTRUCTION),
    (
        "human",
        "User question:\n{question}\n\n"
        "SCHEMA (STRICT):\n"
        "- tables: users, event\n"
        f"- join: {JOIN_RULE}\n"
        "- users columns: id INTEGER PRIMARY KEY, name TEXT NOT NULL\n"
        "- event columns: id INTEGER PRIMARY KEY, protectee_id INTEGER NOT NULL, "
        "timestamp TEXT NOT NULL, ppg_json TEXT, ppg_threat_detected INTEGER, "
        "hrv INTEGER, stress INTEGER, imu_danger_level INTEGER, latitude REAL, "
        "longitude REAL, zone_type TEXT, is_watch_connected INTEGER\n\n"
        "Resolved date (if any): {resolved_date_yyyy_mm_dd}\n"
        "Resolved time window (if any): from {resolved_from} to {resolved_to}\n"
        "Resolved time-of-day filter (if any): op={time_op}, value={time_hhmmss}\n"
        "Return ONLY one valid SQLite SELECT (no commentary)."
    ),
])


def _extract_latest_question(state):
    q = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            q = m.content
            break
    return q


def query_gen_node(state):
    llm = get_chat_llm()
    question = _extract_latest_question(state)
    metric_col = choose_metric(question)

    # (A) 하루 날짜 (선택적으로 제공)
    resolved_date = extract_date_yyyy_mm_dd(question)

    # (B) 이번주/지난주 등 주간 창
    resolved_from = resolved_to = ""
    week_win = resolve_week_window_kst(question)
    if week_win:
        a, b = to_yyyy_mm_dd_hh_mm_ss_strings(week_win)
        resolved_from, resolved_to = a, b

    # (C) "밤 9시 이후" 같은 시각 필터
    time_op = time_hhmmss = ""
    tf = extract_time_filter(question)
    if tf:
        time_op, time_hhmmss = tf  # ('>=', '21:00:00') 등

    prompt = query_gen_prompt.partial(
        question=question,
        metric_col=metric_col,
        resolved_date_yyyy_mm_dd=(resolved_date or ""),
        resolved_from=resolved_from,
        resolved_to=resolved_to,
        time_op=time_op,
        time_hhmmss=time_hhmmss,
    )

    raw = (prompt | llm.bind(
        stop=["\n\n", "/*", "SCHEMA (STRICT):", "CREATE TABLE", "System:", "Human:", "AI:", "Tool:", "```"]
    ) | StrOutputParser()).invoke({})
    text = extract_sql(raw)
    if not text:
        return {"messages": [AIMessage(content="Error: No valid SQL to check")]}
    return {"messages": [AIMessage(content=text)]}



# ==============================
# Query check + execution routing
# ==============================

# NOTE: escape with double braces to avoid ChatPromptTemplate var parsing
query_check_system_json = r"""You are a careful SQLite expert.
Review the given SQL query for common mistakes:
- NOT IN with NULLs
- UNION vs UNION ALL
- BETWEEN used for exclusive ranges
- Type mismatches
- Proper quoting of identifiers
- Wrong function arg counts
- Casting issues
- Wrong join columns

If mistakes exist, rewrite the query; otherwise, return it as-is.

Return ONLY valid JSON, no code fences, no extra text, with schema:
{{"sql": "<final_sql_to_execute>"}}
"""

query_check_prompt = ChatPromptTemplate.from_messages([
    ("system", query_check_system_json),
    ("human", "SQL to check:\n{sql}"),
])

json_parser = JsonOutputParser()
from app.graph.guards import validate_sql_against_schema


def _robust_json_parse(text: str) -> Dict[str, Any]:
    import json, re
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(0)
    data = json.loads(cleaned)
    if not isinstance(data, dict) or "sql" not in data or not isinstance(data["sql"], str):
        raise ValueError("JSON schema invalid: needs {'sql': str}")
    return data

from langchain_core.output_parsers import StrOutputParser as _StrOut
from langchain_core.runnables import RunnableLambda as _RL


# ---------- Helper fixes (멱등/보정) ----------

def _inject_non_null_guards(sql: str, metric_col: str) -> str:
    import re
    cond = "e.timestamp IS NOT NULL AND e.timestamp <> ''"
    # 이미 조건이 포함되어 있으면 재주입하지 않음 (멱등)
    if re.search(r"(?i)e\.timestamp\s+IS\s+NOT\s+NULL", sql) and re.search(r"(?i)e\.timestamp\s*<>\s*''", sql):
        return sql
    parts = re.split(r"(?i)\bORDER\s+BY\b", sql, maxsplit=1)
    head = parts[0].strip()
    tail = ("ORDER BY " + parts[1]) if len(parts) == 2 else ""
    if re.search(r"(?i)\bWHERE\b", head):
        head = re.sub(r"(?i)\bWHERE\b", f"WHERE {cond} AND ", head, count=1)
    else:
        head = head + f" WHERE {cond}"
    return (head + (" " + tail if tail else "")).strip()

def _normalize_time_literal_filters(sql: str) -> str:
    import re
    # e.timestamp >= '21(:00(:00))?' → strftime(...) >= '21:00:00'
    def _fmt(h, m="00", s="00"):
        return f"{int(h):02d}:{int(m or 0):02d}:{int(s or 0):02d}"
    pat_ge = re.compile(r"(e\.timestamp\s*(>=|>)\s*')(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?(')", re.I)
    pat_le = re.compile(r"(e\.timestamp\s*(<=|<)\s*')(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?(')", re.I)
    sql = pat_ge.sub(lambda m: f"strftime('%H:%M:%S', e.timestamp) {m.group(2)} '{_fmt(m.group(3), m.group(4), m.group(5))}'", sql)
    sql = pat_le.sub(lambda m: f"strftime('%H:%M:%S', e.timestamp) {m.group(2)} '{_fmt(m.group(3), m.group(4), m.group(5))}'", sql)
    return sql

def _strip_non_grouped_when_aggregate(sql: str) -> str:
    import re
    if not re.search(r"(?i)\b(AVG|SUM|MIN|MAX|COUNT)\s*\(", sql):
        return sql
    if re.search(r"(?i)\bGROUP\s+BY\b", sql):
        return sql
    m = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql)
    if not m:
        return sql
    select_expr = m.group(1)
    agg = re.search(r"(?i)\b(AVG|SUM|MIN|MAX|COUNT)\s*\([^)]+\)", select_expr)
    if not agg:
        return sql
    return re.sub(r"(?is)^\s*SELECT\s+.*?\s+FROM\s",
                  f"SELECT {agg.group(0)} FROM ", sql)

def _ensure_metric_in_select_for_extremes(sql: str, metric_col: str, want_when: bool, is_extreme: bool) -> str:
    import re
    if not is_extreme:
        return sql
    # 집계/그룹바이는 건드리지 않음
    if re.search(r"(?i)\b(AVG|SUM|COUNT|MIN|MAX)\s*\(", sql) or re.search(r"(?i)\bGROUP\s+BY\b", sql):
        return sql
    # '언제/시각'을 물으면 timestamp+metric을 함께 SELECT
    if want_when:
        return re.sub(r"(?is)^\s*SELECT\s+.*?\s+FROM\s", f"SELECT e.timestamp, e.{metric_col} FROM ", sql)
    # 그렇지 않으면 metric만
    return re.sub(r"(?is)^\s*SELECT\s+.*?\s+FROM\s", f"SELECT e.{metric_col} FROM ", sql)

def _normalize_between_to_half_open(sql: str) -> str:
    import re
    pat = re.compile(r"(e\.timestamp)\s+BETWEEN\s+'([^']+)'\s+AND\s+'([^']+)'", re.I)
    return pat.sub(r"\1 >= '\2' AND \1 < '\3'", sql)

# --- 새로 추가: 불필요한 날짜/시각 필터 제거 (질문에 의도 없으면) ---
_DATE_RANGE_RE = re.compile(r"\s+AND\s+e\.timestamp\s*(>=|>|<=|<)\s*'[^']+'", re.I)
_TOD_RE = re.compile(r"\s+AND\s*strftime\('%H:%M:%S'\s*,\s*e\.timestamp\)\s*(=|>=|>|<=|<)\s*'[^']+'", re.I)
_BETWEEN_RE = re.compile(r"\s+AND\s+e\.timestamp\s+BETWEEN\s+'[^']+'\s+AND\s+'[^']+'", re.I)

def _strip_unwanted_time_filters(sql: str, has_any_time_window: bool) -> str:
    """질문에 시간/날짜 의도가 없으면 쿼리에 끼어든 날짜/시각 필터를 제거."""
    if has_any_time_window:
        return sql
    prev = None
    while prev != sql:
        prev = sql
        sql = _BETWEEN_RE.sub("", sql)
        sql = _DATE_RANGE_RE.sub("", sql)
        sql = _TOD_RE.sub("", sql)
        # 잔여 구두점 정리
        sql = re.sub(r"\bWHERE\s+AND\b", "WHERE ", sql, flags=re.I)
        sql = re.sub(r"\s+AND\s+AND\s+", " AND ", sql, flags=re.I)
        sql = re.sub(r"\s+WHERE\s*$", "", sql, flags=re.I)
    return sql.strip()

def _ensure_order_for_extremes(sql: str, metric_col: str, direction: str | None) -> str:
    """극값 질문인데 ORDER BY가 없으면 추가."""
    if not direction:
        return sql
    dir_kw = "DESC" if direction == "max" else "ASC"
    if re.search(r"(?i)\bORDER\s+BY\b", sql):
        return sql
    sql = sql.strip()
    # LIMIT이 없으면 1개로 한정 (시각을 묻는 ‘가장’ 질문)
    suffix = " LIMIT 1" if "LIMIT" not in sql.upper() else ""
    return f"{sql} ORDER BY e.{metric_col} {dir_kw}, e.timestamp DESC{suffix}"


def _to_min_ts(ts: str) -> str:
    # 'YYYY-MM-DD HH:MM:SS' → 'YYYY-MM-DD HH:MM'
    return ts[:16] if isinstance(ts, str) and len(ts) >= 16 else ts

def _ensure_group_by_for_agg_order(sql: str) -> str:
    import re
    if re.search(r"(?i)\bGROUP\s+BY\b", sql):
        return sql
    # ORDER BY에 집계가 있고, SELECT에 u.name이 있을 때만 주입
    if not re.search(r"(?i)\bORDER\s+BY\s+(?:AVG|SUM|MIN|MAX|COUNT)\s*\(", sql):
        return sql
    m = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql)
    select_expr = m.group(1) if m else ""
    if re.search(r"(?i)\bu\.name\b", select_expr):
        # 첫 ORDER BY 앞에 GROUP BY u.name 삽입
        return re.sub(r"(?i)\bORDER\s+BY\b", "GROUP BY u.name ORDER BY", sql, count=1)
    return sql



def model_check_query(state):
    from app.graph.routing import choose_metric
    from app.core.tools import db_query_tool
    from langchain_core.messages import AIMessage
    from app.core.llm import get_chat_llm

    llm = get_chat_llm()
    candidate_raw = (state["messages"][-1].content or "").strip()
    from app.graph.guards import extract_sql
    candidate_sql = extract_sql(candidate_raw)

    if not candidate_sql or not SQL_HEAD.search(candidate_sql):
        return {"messages": [AIMessage(content="Error: No valid SQL to check")]}

    ok, why = validate_sql_against_schema(candidate_sql)
    if not ok:
        hint = (
            "Use only tables/columns from users(id, name); event(id, protectee_id, timestamp, ppg_json, ppg_threat_detected, hrv, stress, imu_danger_level, latitude, longitude, zone_type, is_watch_connected). "
            f"Join rule: {JOIN_RULE}. Use aliases e (event) and u (users)."
        )
        return {"messages": [AIMessage(content=f"Error: {why}. {hint}")]}

    question = _extract_latest_question(state)
    must_col = choose_metric(question)
    direction = detect_extreme_direction(question)
    want_when = asks_when(question)

    import re as _re
    if not _re.search(rf"\b(?:e\.)?{must_col}\b", candidate_sql, _re.I):
        return {"messages": [AIMessage(content=f"Error: Wrong metric. Use e.{must_col} for this question.")]}  # enforce correct metric

    # LLM self-check (returns {"sql": "..."} as JSON)
    primary_check  = query_check_prompt | llm | json_parser
    fallback_check = query_check_prompt | llm | (_StrOut() | _RL(_robust_json_parse))
    query_check_return_sql = primary_check.with_fallbacks([fallback_check])

    checked = query_check_return_sql.invoke({"sql": candidate_sql})
    final_sql = (checked.get("sql") or "").strip()
    if not final_sql or "<final_sql_to_execute>" in final_sql or not SQL_HEAD.search(final_sql):
        final_sql = candidate_sql

    # 시간/날짜 의도 확인
    has_date = bool(extract_date_yyyy_mm_dd(question))
    has_week = bool(resolve_week_window_kst(question))
    has_time = bool(extract_time_filter(question))
    has_any_time_window = has_date or has_week or has_time

    # ✅ 최소 보정: 의미 왜곡 없이 오류만 예방 + 일관 출력 보장
    final_sql = _normalize_time_literal_filters(final_sql)
    final_sql = _normalize_between_to_half_open(final_sql)
    final_sql = _strip_unwanted_time_filters(final_sql, has_any_time_window)
    final_sql = _inject_non_null_guards(final_sql, must_col)
    final_sql = _strip_non_grouped_when_aggregate(final_sql)
    final_sql = _ensure_group_by_for_agg_order(final_sql)
    final_sql = _ensure_metric_in_select_for_extremes(final_sql, must_col, want_when, bool(direction))
    final_sql = _ensure_order_for_extremes(final_sql, must_col, direction)

    import re as _re2
    if _re2.search(r"(?i)\b(DROP|ALTER|TRUNCATE|ATTACH|DETACH)\b", final_sql):
        return {"messages": [AIMessage(content=f"Error: Refusing to run potentially dangerous SQL: {final_sql}")]}

    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": getattr(db_query_tool, "name", "db_query_tool"),
                        "args": {"query": final_sql},
                        "id": f"run_sql_{uuid.uuid4()}",
                    }
                ],
            )
        ]
    }



def format_answer(state):
    from app.graph.guards import parse_tool_result
    from langchain_core.messages import AIMessage, HumanMessage
    import re as _re

    # ---- Tool 결과 찾기 ----
    tool_msg_content = None
    for m in reversed(state["messages"]):
        if hasattr(m, "name") and m.name == getattr(db_query_tool, "name", "db_query_tool"):
            tool_msg_content = m.content
            break
    if tool_msg_content is None:
        return {"messages": [AIMessage(content="Error: No tool result found")]}

    ok, payload = parse_tool_result(tool_msg_content)
    if not ok:
        return {"messages": [AIMessage(content=payload)]}

    # 문자열이면 그대로 반환
    if isinstance(payload, str):
        return {"messages": [AIMessage(content=f"Answer: {payload}")]}

    rows = payload
    if not rows:
        return {"messages": [AIMessage(content="Answer: 결과가 비어 있습니다.")]}

    only = rows[0]

    # ---- 유틸 ----
    TS_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")
    def _is_ts(x): 
        return isinstance(x, str) and TS_RE.match(x) is not None

    def _is_num(x):
        try:
            return isinstance(x, (int, float)) and x is not True and x is not False
        except Exception:
            return False

    def _fmt_num(x, digits=1):
        """숫자를 'digits' 자리로 반올림하고 불필요한 0/소수점 제거"""
        try:
            s = f"{float(x):.{digits}f}"
            s = s.rstrip('0').rstrip('.')
            return s
        except Exception:
            return x

    def _fmt_any(x, digits=1):
        return _fmt_num(x, digits) if _is_num(x) else x

    # 질문 의도(최대/최소) 파악
    question = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            question = m.content or ""
            break
    direction = detect_extreme_direction(question)  # "max" | "min" | None

    MAX_SHOW = 10

    # ---- Case A) (timestamp, numeric): 시각 + 값 묶음 ----
    if isinstance(only, (list, tuple)) and len(only) >= 2 and _is_ts(only[0]) and _is_num(only[1]):
        vals = [(r[0], r[1]) for r in rows
                if isinstance(r, (list, tuple)) and len(r) >= 2 and _is_ts(r[0]) and _is_num(r[1])]
        if not vals:
            return {"messages": [AIMessage(content="Answer: 결과가 비어 있습니다.")]}

        if direction == "min":
            target = min(v for _, v in vals)
        else:
            target = max(v for _, v in vals)

        ties = [(ts, v) for ts, v in vals if v == target]
        if len(ties) == 1:
            ts, val = ties[0]
            return {"messages": [AIMessage(content=f"Answer: {_to_min_ts(ts)} (지수 {_fmt_num(val, 1)})")]}

        shown = ties[:MAX_SHOW]
        rest = len(ties) - len(shown)
        bullets = "\n".join(f"- {_to_min_ts(ts)} (지수 {_fmt_num(v, 1)})" for ts, v in shown)
        suffix = "" if rest <= 0 else f"\n(+{rest}개 더)"
        return {"messages": [AIMessage(content=f"Answer:\n{bullets}{suffix}")]}

    # ---- Case B) (timestamp) 단독: 시각 나열 ----
    if isinstance(only, (list, tuple)) and len(only) == 1 and _is_ts(only[0]):
        ts_list = []
        for r in rows:
            if isinstance(r, (list, tuple)) and len(r) == 1 and _is_ts(r[0]):
                ts_list.append(r[0][:16])  # YYYY-MM-DD HH:MM
        seen = set(); dedup = []
        for t in ts_list:
            if t not in seen:
                seen.add(t); dedup.append(t)
        if not dedup:
            return {"messages": [AIMessage(content="Answer: 결과가 비어 있습니다.")]}

        bullets = "\n".join(f"- {t}" for t in dedup[:MAX_SHOW])
        rest = len(dedup) - min(len(dedup), MAX_SHOW)
        suffix = "" if rest <= 0 else f"\n(+{rest}개 더)"
        return {"messages": [AIMessage(content=f"Answer:\n{bullets}{suffix}")]}

    # ---- Case C) 그 외: 첫 컬럼 위주 안전 출력 (숫자는 반올림 표시) ----
    values = []
    if isinstance(rows, (list, tuple)):
        for r in rows:
            if isinstance(r, (list, tuple)):
                values.append(r[0] if len(r) > 0 else r)
            else:
                values.append(r)
    else:
        values = [rows]

    if len(values) == 1:
        v = values[0]
        # AVG 결과가 None 등일 때 처리
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return {"messages": [AIMessage(content="Answer: 결과가 비어 있습니다.")]}
        if _is_num(v):
            v = _fmt_num(v, 1)  # 🔹 단일 숫자는 기본 1자리로
        out = f"Answer: {v}"
        return {"messages": [AIMessage(content=out)]}

    SHOWN = min(len(values), 10)
    bullets = "\n".join(f"- {_fmt_any(values[i], 1)}" for i in range(SHOWN))
    suffix = "" if SHOWN == len(values) else f"\n(+{len(values)-SHOWN}개 더)"
    return {"messages": [AIMessage(content=f"Answer:\n{bullets}{suffix}")]}




def after_answer(state):
    text = (state["messages"][-1].content or "").strip()
    if text.startswith("Error:"):
        return "query_gen"
    if text.startswith("Answer:"):
        return "narrate_answer"  
    from langgraph.graph import END
    return END


def should_continue(state):
    text = (state["messages"][-1].content or "").strip()
    if ANSWER_SQL.match(text):
        return "correct_query"
    if SQL_HEAD.search(text):
        return "correct_query"
    if text.startswith("Answer:"):
        from langgraph.graph import END
        return END
    if text.startswith("Error:"):
        return "query_gen"
    return "correct_query"


def route_after_check(state):
    last = state["messages"][-1]
    from langchain_core.messages import AIMessage
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execute_query"
    return "query_gen"
