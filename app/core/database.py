from langchain_community.utilities import SQLDatabase
from app.config import settings

_db_singleton: SQLDatabase | None = None

def get_db() -> SQLDatabase:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = SQLDatabase.from_uri(settings.sqlite_uri)
    return _db_singleton