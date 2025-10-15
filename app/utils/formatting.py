import re

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")

def is_ts(x) -> bool:
    return isinstance(x, str) and _TS_RE.match(x) is not None

def is_num(x) -> bool:
    try:
        return isinstance(x, (int, float)) and x is not True and x is not False
    except Exception:
        return False

def fmt_num(x, digits: int = 1):
    try:
        s = f"{float(x):.{digits}f}"
        s = s.rstrip('0').rstrip('.')
        return s
    except Exception:
        return x

def fmt_any(x, digits: int = 1):
    return fmt_num(x, digits) if is_num(x) else x

def to_min_ts(ts: str) -> str:
    # 'YYYY-MM-DD HH:MM:SS' → 'YYYY-MM-DD HH:MM'
    return ts[:16] if isinstance(ts, str) and len(ts) >= 16 else ts
