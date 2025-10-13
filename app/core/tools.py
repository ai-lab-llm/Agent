from langchain_core.tools import tool
from app.core.database import get_db

@tool
def db_query_tool(query: str) -> str:
    """
    Run SQL queries against the SQLite database and return results.
    Returns an error string if the query failed.
    """
    db = get_db()
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result