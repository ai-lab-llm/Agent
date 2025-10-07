from datetime import datetime, timedelta, timezone
import re
KST = timezone(timedelta(hours=9))      # UTC+9 (한국 표준시)

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