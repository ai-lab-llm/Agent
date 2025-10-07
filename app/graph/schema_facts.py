from langchain_core.messages import AIMessage

JOIN_RULE = "event.protectee_id = users.id"

SCHEMA_STRICT_TEXT = (
    "SCHEMA (STRICT):\n"
    "- tables: users, event\n"
    f"- join: {JOIN_RULE}\n"
    "- users columns: id INTEGER PRIMARY KEY, name TEXT NOT NULL\n"
    "- event columns: id INTEGER PRIMARY KEY, protectee_id INTEGER NOT NULL, "
    "timestamp TEXT NOT NULL, ppg_json TEXT, ppg_threat_detected INTEGER, "
    "hrv INTEGER, stress INTEGER, imu_danger_level INTEGER, latitude REAL, "
    "longitude REAL, zone_type TEXT, is_watch_connected INTEGER\n\n"
    "COLUMN NOTES (DO NOT INVENT NEW COLUMNS):\n"
    "- users.id: 사용자 PK (INTEGER)\n"
    "- users.name: 사용자 이름 (TEXT)\n"
    "- event.id: 이벤트 PK (INTEGER)\n"
    "- event.protectee_id: users.id FK (INTEGER)\n"
    "- event.timestamp: 이벤트 시각 (TEXT, 'YYYY-MM-DD HH:MM:SS')\n"
    "- event.ppg_json: PPG 값 배열(JSON 문자열)\n"
    "- event.ppg_threat_detected: PPG 기반 위협 퍼센트(정수, %)\n"
    "- event.hrv: HRV 지수(정수)\n"
    "- event.stress: 스트레스 지수(정수)\n"
    "- event.imu_danger_level: 움직임/자세 불안정 지표(정수)\n"
    "- event.latitude: 위도(REAL)\n"
    "- event.longitude: 경도(REAL)\n"
    "- event.zone_type: 구역 종류('safe' 또는 'unfamiliar')\n"
    "- event.is_watch_connected: 워치 연결 상태(1=연결, 0=끊김)\n\n"
    "SEMANTICS:\n"
    "- e.stress: 심리·생리적 스트레스 강도(클수록 높음)\n"
    "- e.hrv: 심박 변이 지수(HRV)\n"
    "- e.imu_danger_level: 움직임/균형 불안정, 넘어짐/흔들림 위험도(클수록 높음)\n\n"
    "- e.ppg_threat_detected: PPG 센서 기반 생체신호 위협도(0~100%, 클수록 위험/불안정)\n\n"
    "RULES:\n"
    "- Use ONLY these tables/columns. NEVER invent tables or columns.\n"
    "- Always fully-qualify columns using aliases: event AS e, users AS u.\n"
    "- Filter by user name via JOIN (u.id = e.protectee_id AND u.name = ...).\n"
    "- For 'highest/최고/가장 높', ORDER BY the TARGET METRIC column DESC, then e.timestamp DESC.\n"
    "- For '낯선 장소/unfamiliar' add WHERE e.zone_type = 'unfamiliar'; for 'safe' add WHERE e.zone_type='safe'.\n"
    "- Output ONLY a single SQLite SELECT (no DDL/DML, no commentary).\n"
)

def inject_schema_facts(_: dict) -> dict:
    return {"messages": [AIMessage(content=SCHEMA_STRICT_TEXT)]}