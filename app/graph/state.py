from typing import Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]