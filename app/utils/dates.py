from __future__ import annotations
from datetime import datetime, timedelta, timezone
import re

# KST (UTC+9)
KST = timezone(timedelta(hours=9))  # 한국 표준시

# 기본 날짜 추출
def extract_date_yyyy_mm_dd(q: str, now: datetime | None = None) -> str | None:
    if now is None:
        now = datetime.now(KST)
    s = q.strip()

    # 상대 날짜
    if "오늘" in s:
        return now.strftime("%Y-%m-%d")
    if "어제" in s:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    if "내일" in s:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*일\s*전", s)
    if m:
        return (now - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*일\s*후", s)
    if m:
        return (now + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")

    # 'M월 D일'
    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
    if m:
        mm = int(m.group(1)); dd = int(m.group(2))
        return f"{now.year:04d}-{mm:02d}-{dd:02d}"

    # 'D일' (현재 월로 가정)
    for m in re.finditer(r"(\d{1,2})\s*일", s):
        dd = int(m.group(1))
        start = m.start()
        j = start - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        if j >= 0 and s[j] == "월":
            continue
        return f"{now.year:04d}-{now.month:02d}-{dd:02d}"

    return None


def _to_kst(dt: datetime) -> datetime:
    """입력 datetime을 KST 타임존의 aware datetime으로 변환."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)

def _fmt_ts(dt: datetime) -> str:
    """'YYYY-MM-DD HH:MM:SS'"""
    return _to_kst(dt).strftime("%Y-%m-%d %H:%M:%S")

def start_of_day_kst(dt: datetime) -> datetime:
    d = _to_kst(dt)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def start_of_week_kst(dt: datetime) -> datetime:
    """
    KST 기준 '월요일 00:00:00'을 주 시작으로 간주.
    (월=0 ... 일=6)
    """
    d = _to_kst(dt)
    monday = d - timedelta(days=d.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)

def week_window_kst(dt: datetime, *, offset_weeks: int = 0) -> tuple[datetime, datetime]:
    """
    dt가 속한 주(월~일)의 [start, end) 구간을 KST 기준으로 반환.
    offset_weeks=-1 이면 지난주, 0이면 이번주, +1이면 다음주.
    """
    start = start_of_week_kst(dt) + timedelta(weeks=offset_weeks)
    end = start + timedelta(days=7)
    return start, end


_WEEK_PATTERNS = [
    (re.compile(r"(이번\s*주|이번주)"), 0),
    (re.compile(r"(지난\s*주|지난주)"), -1),
    (re.compile(r"(다음\s*주|다음주)"), +1),
]

def resolve_week_window_kst(text: str, now: datetime | None = None) -> tuple[datetime, datetime] | None:
    """
    질문에서 '이번주/지난주/다음주'를 인지해 KST 기준 (월~일) [start, end) 구간 반환.
    매칭 실패 시 None.
    """
    if now is None:
        now = datetime.now(KST)
    s = text.strip()
    for pat, off in _WEEK_PATTERNS:
        if pat.search(s):
            return week_window_kst(now, offset_weeks=off)
    return None

def resolve_week_window_strings(text: str, now: datetime | None = None) -> tuple[str, str] | None:
    win = resolve_week_window_kst(text, now=now)
    if not win:
        return None
    a, b = win
    return _fmt_ts(a), _fmt_ts(b)


_PART_OF_DAY = {
    "오전": "am",
    "새벽": "am",
    "오후": "pm",
    "저녁": "pm",
    "밤": "pm",
}

_TIME_PHRASE_RE = re.compile(
    r"(오전|오후|저녁|밤|새벽)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?",
    re.I
)

def _to_24h(hour: int, label: str | None) -> int:
    """
    '오전/오후/저녁/밤/새벽' 라벨을 고려해 24시간제로 보정.
    - am: 12시는 0시 처리
    - pm: 1~11시는 +12, 12시는 12 유지
    - 라벨 없으면 그대로 (이미 0~23 가정)
    """
    if not label:
        return hour
    lab = _PART_OF_DAY.get(label, None)
    if lab == "am":
        return 0 if hour == 12 else hour
    if lab == "pm":
        return 12 if hour == 12 else (hour + 12 if hour < 12 else hour)
    return hour

def extract_time_phrase_hhmmss(text: str) -> str | None:
    """
    텍스트에서 첫 번째 '오전/오후/저녁/밤/새벽 + ?시 + ?분' 패턴을 찾아 'HH:MM:SS'로 반환.
    - 라벨 없고 '21시'처럼 0~23시도 허용.
    - 분 미지정 시 00으로 간주.
    """
    m = _TIME_PHRASE_RE.search(text)
    if not m:
        return None
    label = m.group(1)
    hour = int(m.group(2))
    minute = int(m.group(3) or 0)
    hour24 = _to_24h(hour, label)
    hour24 = max(0, min(23, hour24))
    minute = max(0, min(59, minute))
    return f"{hour24:02d}:{minute:02d}:00"


_TIME_FILTER_RE = re.compile(
    r"((?:오전|오후|저녁|밤|새벽)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?)\s*(이후|이전|부터|까지|이상|이하|초과|미만)",
    re.I
)

_OP_MAP = {
    "이후": ">=",
    "부터": ">=",
    "이상": ">=",
    "초과": ">",

    "이전": "<=",
    "까지": "<=",
    "이하": "<=",
    "미만": "<",
}

def extract_time_filter(text: str) -> tuple[str, str] | None:
    """
    시간 비교 연산자를 추출.
    반환: (op, 'HH:MM:SS')  예: ('>=', '21:00:00')
    매칭 실패 시 None.
    """
    m = _TIME_FILTER_RE.search(text)
    if not m:
        return None
    phrase = m.group(1)
    kw = m.group(2)
    hhmmss = extract_time_phrase_hhmmss(phrase)
    if not hhmmss:
        return None
    op = _OP_MAP.get(kw, None)
    if not op:
        return None
    return op, hhmmss


def resolve_day_window_kst(text: str, now: datetime | None = None) -> tuple[datetime, datetime] | None:
    """
    '오늘/어제/내일'을 [start, end) 구간으로 반환. 없으면 None.
    """
    if now is None:
        now = datetime.now(KST)
    s = text.strip()
    base = None
    if "오늘" in s:
        base = now
    elif "어제" in s:
        base = now - timedelta(days=1)
    elif "내일" in s:
        base = now + timedelta(days=1)
    else:
        return None
    start = start_of_day_kst(base)
    end = start + timedelta(days=1)
    return start, end


def to_yyyy_mm_dd_hh_mm_ss_strings(win: tuple[datetime, datetime]) -> tuple[str, str]:
    """
    (start_dt, end_dt) → ('YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS')
    """
    a, b = win
    return _fmt_ts(a), _fmt_ts(b)
