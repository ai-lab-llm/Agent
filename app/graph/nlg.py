from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_chat_llm
from app.graph.guards import parse_tool_result  
import re as _re

SYSTEM = """당신은 간결한 한국어 문장으로 답변을 정리하는 비서입니다.
규칙:
- 오직 제공된 '원 질문', '정답 문자열', 'rows 요약'만을 근거로 답하세요. 새로운 사실을 만들지 마세요.
- 질문에 '시각/시간/언제'가 포함되었는데 rows 안에 타임스탬프가 없으면, '최대/최소 값은 X이며 시각 정보는 결과에 포함되지 않았습니다'처럼 안전하게 표현하세요.
- rows가 (timestamp, value) 쌍이면 시각과 값을 함께 한 문장으로 말하세요.
- 숫자만 있을 때는 '값'임을 명시하세요(예: '최대 스트레스 값은 91입니다').
- rows가 (timestamp, value) 쌍이면 반드시 날짜와 시각(분까지)과 값을 함께 자연스럽게 말하세요. (예: "… 19시 50분, 값은 99로 가장 높았습니다")
- ISO 타임스탬프(YYYY-MM-DD HH:MM:SS)는 한국어 표기로 바꿔도 됩니다(예: 2025년 8월 18일 17시 30분).
- 여러 항목이 열거된 경우 최대 3개까지만 언급하고, 나머지는 '외 N건'으로 축약하세요.
- 출력은 반드시 'Final: '로 시작하세요.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "원 질문:\n{question}\n\n"
     "정답 문자열:\n{answer}\n\n"
     "rows 요약:\n{rows_summary}\n"
     "타임스탬프_존재: {has_ts}\n"
     "숫자값_존재: {has_val}\n\n"
     "위 정보를 바탕으로 Final: 로 시작하는 한두 문장을 작성하세요.")
])

TS_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")

def _last_user_question(state) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""

def _last_answer_line(state) -> str | None:
    for m in reversed(state["messages"]):
        if hasattr(m, "type") and m.type == "ai":
            c = (getattr(m, "content", "") or "").strip()
            if isinstance(c, str) and c.startswith("Answer:"):
                return c
    return None

def _last_tool_rows(state):
    for m in reversed(state["messages"]):
        if hasattr(m, "name") and m.name == "db_query_tool":
            ok, payload = parse_tool_result(m.content or "")
            if ok:
                return payload
    return None

def _summarize_rows(rows, limit=3):
    try:
        import json
        s = json.dumps(rows, ensure_ascii=False)
        if len(s) > 800:
            s = s[:800] + f"...(총 {len(s)}자 중 앞 800자만)"
        return s
    except Exception:
        return str(rows)[:800]

def _scan_has_ts_val(rows):
    has_ts = False
    has_val = False
    if isinstance(rows, (list, tuple)) and rows:
        r0 = rows[0]
        if isinstance(r0, (list, tuple)):
            if len(r0) >= 1 and isinstance(r0[0], str) and TS_RE.match(r0[0]):
                has_ts = True
            if len(r0) >= 2 and isinstance(r0[1], (int, float)):
                has_val = True
        else:
            has_val = isinstance(r0, (int, float))
            has_ts = isinstance(r0, str) and TS_RE.match(r0) is not None
    return has_ts, has_val

def narrate_answer(state):
    question = _last_user_question(state)
    answer = _last_answer_line(state)
    if not answer:
        return {"messages": [AIMessage(content="Final: 조건에 맞는 결과가 없습니다.")]}

    rows = _last_tool_rows(state)
    rows_summary = _summarize_rows(rows) if rows is not None else "(없음)"
    has_ts, has_val = _scan_has_ts_val(rows) if rows is not None else (False, False)

    llm = get_chat_llm()
    text = (PROMPT | llm | StrOutputParser()).invoke({
        "question": question,
        "answer": answer,
        "rows_summary": rows_summary,
        "has_ts": "예" if has_ts else "아니오",
        "has_val": "예" if has_val else "아니오",
    }).strip()

    if not text.startswith("Final:"):
        text = "Final: " + text
    return {"messages": [AIMessage(content=text)]}
