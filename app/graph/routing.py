import json, re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from app.core.llm import get_chat_llm

METRIC_TO_COL = {
    "imu_danger_level": "imu_danger_level",
    "stress": "stress",
    "hrv": "hrv",
    "ppg_threat_detected": "ppg_threat_detected",
}

# === 의미 기반 스코어링(0~100)으로 단일 메트릭 선택: few-shot/키워드 폴백 없음 ===
_SCORING_SYSTEM = """You are a strict semantic router for Korean analytics questions.
Score how relevant EACH metric is (0~100; integers) by MEANING (not keywords). Then pick exactly ONE best metric.

Canonical meanings:
- imu_danger_level: bodily movement/posture/balance instability; shaking/tremor; fall-risk (physical instability of the person’s body)
- stress: psychological/physiological stress state (mental/physio burden; NOT a sensor signal)
- hrv: the HRV metric itself (variation of heartbeat intervals; a sensor-derived physiological metric)
- ppg_threat_detected: PPG-based biosignal overall threat percent (sensor-level anomaly score; a summary for biosignals)

Disambiguation (VERY IMPORTANT):
- If the subject of "instability(불안정)" is the **person's movement/body/posture/balance**, choose imu_danger_level.
- If the subject is the **biosignal/sensor readings themselves** (e.g., “생체신호/신호/센서 수치가 불안정/이상”), do NOT choose imu_danger_level.
  Prefer ppg_threat_detected as the overall biosignal threat; choose hrv ONLY when HRV is explicitly the target.
- If the question explicitly names a metric (stress/HRV/PPG/IMU), choose that metric.
- Resolve ties by these priorities (from generic to specific):
  biosignal-overall → ppg_threat_detected; explicit HRV → hrv; motion/body instability → imu_danger_level; mental state → stress.

Output ONLY compact JSON (no code fences, no extra text):
{{
  "scores": {{"imu_danger_level": <0-100>, "stress": <0-100>, "hrv": <0-100>, "ppg_threat_detected": <0-100>}},
  "metric": "<imu_danger_level|stress|hrv|ppg_threat_detected>"
}}
"""

_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SCORING_SYSTEM),
    ("human", "Question: {question}\nReturn JSON only.")
])

def _parse_scores(text: str):
    s = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m: s = m.group(0)
    data = json.loads(s)

    scores = data.get("scores") or {}
    norm = {}
    for k in METRIC_TO_COL.keys():
        try:
            norm[k] = int(scores.get(k, 0))
        except Exception:
            try: norm[k] = int(float(scores.get(k, 0)))
            except Exception: norm[k] = 0

    metric = (data.get("metric") or "").strip()
    if metric not in METRIC_TO_COL:
        metric = max(norm, key=norm.get)  # argmax fallback
    return metric, norm

def choose_metric(question: str) -> str:
    llm = get_chat_llm()
    parser = StrOutputParser() | RunnableLambda(_parse_scores)
    metric_label, _scores = (_SCORING_PROMPT | llm | parser).invoke({"question": question})
    return METRIC_TO_COL[metric_label]
