import json, re
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from app.core.llm import get_chat_llm

METRIC_TO_COL = {
    "imu_danger_level": "imu_danger_level", 
    "stress": "stress",                  
    "hrv": "hrv",
    "ppg_threat_detected": "ppg_threat_detected",                
}

SEMANTICS_TEXT = (
    "Column meanings:\n"
    "- imu_danger_level: movement instability / fall / shaking / balance risk\n"
    "- stress: psychological/physiological stress level\n"
    "- hrv: heart rate variability index\n"
    "- ppg_threat_detected: PPG-based biosignal threat percent (0~100; higher=worse)\n"
)

metric_classify_system = """You are a strict router.
Classify the user's question into EXACTLY one metric among:
- imu_danger_level (movement instability/fall/shaking/balance risk)
- stress (psychological/physiological stress)
- hrv (heart rate variability)
- ppg_threat_detected (PPG-based biosignal threat; percent; higher is more dangerous)

Rules:
- Output ONLY JSON, no text, with schema: {"metric": "<one_of_above>"}
- Prefer imu_danger_level when the question is about movement, balance, shaking, fall, instability, posture, acceleration, or '움직임 위험도'.
- Prefer stress when explicitly about stress/스트레스.
- Prefer hrv when about HRV/심박변이.
- Prefer ppg_threat_detected when about PPG/생체신호 위협/위험 퍼센트/신호 이상.
"""

metric_classify_prompt = ChatPromptTemplate.from_messages([
    ("system", metric_classify_system),
    ("human", "Question:\n{question}\n\n{semantics}\nReturn JSON only."),
])

json_parser = JsonOutputParser()


def _metric_robust(text: str):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(0)
    data = json.loads(cleaned)
    if not isinstance(data, dict) or "metric" not in data:
        raise ValueError("bad json")
    return data

metric_robust_parser = StrOutputParser() | RunnableLambda(_metric_robust)


def choose_metric(question: str) -> str:
    llm = get_chat_llm()
    metric_router = (metric_classify_prompt | llm | json_parser).with_fallbacks(
        [metric_classify_prompt | llm | metric_robust_parser]
    )
    try:
        out = metric_router.invoke({"question": question, "semantics": SEMANTICS_TEXT})
        metric = (out.get("metric") or "").strip()
        col = METRIC_TO_COL.get(metric)
        if col:
            return col
    except Exception:
        pass
    q = question.lower()
    if any(k in q for k in ["ppg", "위협", "생체신호", "threat"]):
        return "ppg_threat_detected"
    if any(k in q for k in ["움직임 불안정","움직임 위험도","균형","넘어짐","흔들림","자세","가속도","movement","fall","shake","balance"]):
        return "imu_danger_level"
    if "hrv" in q or "심박" in q:
        return "hrv"
    return "stress"