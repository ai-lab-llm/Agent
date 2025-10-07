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
from app.utils.dates import extract_date_yyyy_mm_dd

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


# Query generation
QUERY_GEN_INSTRUCTION = """You are a SQL expert.

YOU MUST follow these constraints strictly:
- Use ONLY tables/columns explicitly listed in the SCHEMA (STRICT) message below.
- NEVER invent tables or columns. If something is missing, output: Error: Missing data
- Always fully-qualify columns with aliases: event AS e, users AS u.
- To filter by a user name, JOIN users u ON u.id = e.protectee_id and filter u.name = '<name>'.
- e.timestamp is a TEXT datetime. Use strftime only if formatting is needed.
- Treat {metric_col} as the TARGET METRIC for this question.
- For 'highest/최고/가장 높', ORDER BY e.{metric_col} DESC, then e.timestamp DESC.
- If the question mentions '낯선 장소', '낯선 구역', or 'unfamiliar', include WHERE e.zone_type = 'unfamiliar'.
- If it mentions '안전 구역' or 'safe', include WHERE e.zone_type = 'safe'.
- When the user asks for "time/시각/시간/언제/언제였어/언제인가", ALWAYS include e.timestamp in the SELECT along with the target metric.
- If user explicitly asks only for the metric value, you may return only the metric. Otherwise, prefer SELECT e.timestamp, e.{metric_col}.
- Output ONLY a single valid SQLite SELECT (no backticks, no explanation). No DDL/DML statements.

If a query was executed successfully and the result is sufficient, output:
Answer: <concise answer>

If the immediately previous step shows an execution error, FIX the query and output only the corrected SQL.
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
    resolved_date = extract_date_yyyy_mm_dd(question)
    prompt = query_gen_prompt.partial(
        question=question,
        metric_col=metric_col,
        resolved_date_yyyy_mm_dd=(resolved_date or "")
    )
    raw = (prompt | llm.bind(
        stop=["\n\n", "/*", "SCHEMA (STRICT):", "CREATE TABLE", "System:", "Human:", "AI:", "Tool:", "```"]
    ) | StrOutputParser()).invoke({})
    text = extract_sql(raw)
    if not text:
        return {"messages": [AIMessage(content="Error: No valid SQL to check")]}
    return {"messages": [AIMessage(content=text)]}


# Query check + execution routing
query_check_system_json = """You are a careful SQLite expert.
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

    import re as _re
    if not _re.search(rf"\b(?:e\.)?{must_col}\b", candidate_sql, _re.I):
        return {"messages": [AIMessage(content=f"Error: Wrong metric. Use e.{must_col} for this question.")]}

    primary_check  = query_check_prompt | llm | json_parser
    fallback_check = query_check_prompt | llm | (_StrOut() | _RL(_robust_json_parse))
    query_check_return_sql = primary_check.with_fallbacks([fallback_check])

    checked = query_check_return_sql.invoke({"sql": candidate_sql})
    final_sql = (checked.get("sql") or "").strip()
    if not final_sql or "<final_sql_to_execute>" in final_sql or not SQL_HEAD.search(final_sql):
        final_sql = candidate_sql

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
    from langchain_core.messages import AIMessage
    import re as _re

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

    # 원시 문자열이면 그대로
    if isinstance(payload, str):
        return {"messages": [AIMessage(content=f"Answer: {payload}")]}

    rows = payload
    if not rows:
        return {"messages": [AIMessage(content="Answer: 결과가 비어 있습니다.")]}
    only = rows[0]

    TS_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")
    def _is_ts(x): return isinstance(x, str) and TS_RE.match(x) is not None
    def _is_num(x):
        try: return isinstance(x, (int, float))
        except Exception: return False

    MAX_SHOW = 10

    # Case A) (timestamp, numeric) 형태: 시각과 값 함께 요약
    if isinstance(only, (list, tuple)) and len(only) >= 2 and _is_ts(only[0]) and _is_num(only[1]):
        max_val = rows[0][1]
        ties = [(r[0], r[1]) for r in rows
                if isinstance(r, (list, tuple)) and len(r) >= 2 and _is_ts(r[0]) and _is_num(r[1]) and r[1] == max_val]
        if len(ties) == 1:
            ts, val = ties[0]
            return {"messages": [AIMessage(content=f"Answer: {ts} (지수 {val})")]}
        shown = ties[:MAX_SHOW]
        rest = len(ties) - len(shown)
        bullets = "\n".join(f"- {ts} (지수 {val})" for ts, val in shown)
        suffix = "" if rest <= 0 else f"\n(+{rest}개 더)"
        return {"messages": [AIMessage(content=f"Answer:\n{bullets}{suffix}")]} 

    # Case B) (timestamp) 단독
    if isinstance(only, (list, tuple)) and len(only) == 1 and _is_ts(only[0]):
        return {"messages": [AIMessage(content=f"Answer: {only[0]}")]}

    if _is_ts(only):
        return {"messages": [AIMessage(content=f"Answer: {only}")]}

    # Case C) 그 외: 첫 컬럼들만 안전하게 나열
    values: list = []
    if isinstance(rows, (list, tuple)):
        for r in rows:
            if isinstance(r, (list, tuple)):
                values.append(r[0] if len(r) > 0 else r)
            else:
                values.append(r)
    else:
        values = [rows]

    if len(values) == 1:
        out = f"Answer: {values[0]}"
    else:
        SHOWN = min(len(values), 10)
        bullets = "\n".join(f"- {values[i]}" for i in range(SHOWN))
        suffix = "" if SHOWN == len(values) else f"\n(+{len(values)-SHOWN}개 더)"
        out = f"Answer:\n{bullets}{suffix}"

    return {"messages": [AIMessage(content=out)]}



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