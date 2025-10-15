import re

# 날짜/시각 필터 정규식
_DATE_RANGE_RE = re.compile(r"\s+AND\s+e\.timestamp\s*(>=|>|<=|<)\s*'[^']+'", re.I)
_TOD_RE = re.compile(r"\s+AND\s*strftime\('%H:%M:%S'\s*,\s*e\.timestamp\)\s*(=|>=|>|<=|<)\s*'[^']+'", re.I)
_BETWEEN_RE = re.compile(r"\s+AND\s+e\.timestamp\s+BETWEEN\s+'[^']+'\s+AND\s+'[^']+'", re.I)

def normalize_time_literal_filters(sql: str) -> str:
    def _fmt(h, m="00", s="00"):
        return f"{int(h):02d}:{int(m or 0):02d}:{int(s or 0):02d}"
    pat_ge = re.compile(r"(e\.timestamp\s*(>=|>)\s*')(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?(')", re.I)
    pat_le = re.compile(r"(e\.timestamp\s*(<=|<)\s*')(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?(')", re.I)
    sql = pat_ge.sub(lambda m: f"strftime('%H:%M:%S', e.timestamp) {m.group(2)} '{_fmt(m.group(3), m.group(4), m.group(5))}'", sql)
    sql = pat_le.sub(lambda m: f"strftime('%H:%M:%S', e.timestamp) {m.group(2)} '{_fmt(m.group(3), m.group(4), m.group(5))}'", sql)
    return sql

def normalize_between_to_half_open(sql: str) -> str:
    pat = re.compile(r"(e\.timestamp)\s+BETWEEN\s+'([^']+)'\s+AND\s+'([^']+)'", re.I)
    return pat.sub(r"\1 >= '\2' AND \1 < '\3'", sql)

def strip_unwanted_time_filters(sql: str, has_any_time_window: bool) -> str:
    """질문에 시간/날짜 의도가 없으면 쿼리에 끼어든 날짜/시각 필터 제거."""
    if has_any_time_window:
        return sql
    prev = None
    while prev != sql:
        prev = sql
        sql = _BETWEEN_RE.sub("", sql)
        sql = _DATE_RANGE_RE.sub("", sql)
        sql = _TOD_RE.sub("", sql)
        sql = re.sub(r"\bWHERE\s+AND\b", "WHERE ", sql, flags=re.I)
        sql = re.sub(r"\s+AND\s+AND\s+", " AND ", sql, flags=re.I)
        sql = re.sub(r"\s+WHERE\s*$", "", sql, flags=re.I)
    return sql.strip()

def inject_non_null_guards(sql: str, metric_col: str | None = None) -> str:
    cond = "e.timestamp IS NOT NULL AND e.timestamp <> ''"
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

def strip_non_grouped_when_aggregate(sql: str) -> str:
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
    return re.sub(r"(?is)^\s*SELECT\s+.*?\s+FROM\s", f"SELECT {agg.group(0)} FROM ", sql)

def ensure_metric_in_select_for_extremes(sql: str, metric_col: str, want_when: bool, is_extreme: bool) -> str:
    if not is_extreme:
        return sql
    if re.search(r"(?i)\b(AVG|SUM|COUNT|MIN|MAX)\s*\(", sql) or re.search(r"(?i)\bGROUP\s+BY\b", sql):
        return sql
    if want_when:
        return re.sub(r"(?is)^\s*SELECT\s+.*?\s+FROM\s", f"SELECT e.timestamp, e.{metric_col} FROM ", sql)
    return re.sub(r"(?is)^\s*SELECT\s+.*?\s+FROM\s", f"SELECT e.{metric_col} FROM ", sql)

def ensure_order_for_extremes(sql: str, metric_col: str, direction: str | None) -> str:
    if not direction:
        return sql
    dir_kw = "DESC" if direction == "max" else "ASC"
    if re.search(r"(?i)\bORDER\s+BY\b", sql):
        return sql
    sql = sql.strip()
    suffix = " LIMIT 1" if "LIMIT" not in sql.upper() else ""
    return f"{sql} ORDER BY e.{metric_col} {dir_kw}, e.timestamp DESC{suffix}"

def ensure_group_by_for_agg_order(sql: str) -> str:
    if re.search(r"(?i)\bGROUP\s+BY\b", sql):
        return sql
    if not re.search(r"(?i)\bORDER\s+BY\s+(?:AVG|SUM|MIN|MAX|COUNT)\s*\(", sql):
        return sql
    m = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql)
    select_expr = m.group(1) if m else ""
    if re.search(r"(?i)\bu\.name\b", select_expr):
        return re.sub(r"(?i)\bORDER\s+BY\b", "GROUP BY u.name ORDER BY", sql, count=1)
    return sql

# 이름(그룹)별 평균 비교
def ensure_select_avg_and_drop_limit_for_group_compare(sql: str, metric_col: str) -> str:
    if not re.search(r"(?i)\bGROUP\s+BY\s+u\.name\b", sql):
        return sql
    if not re.search(r"(?i)\bORDER\s+BY\s+AVG\s*\(", sql):
        return sql
    # SELECT 절 교체
    sql = re.sub(
        r"(?is)^\s*SELECT\s+.*?\s+FROM\s",
        f"SELECT u.name, ROUND(AVG(e.{metric_col}), 1) AS avg_val FROM ",
        sql,
        count=1,
    )
    # LIMIT 제거
    sql = re.sub(r"(?i)\s+LIMIT\s+\d+\s*;?\s*$", "", sql).strip()
    return sql
