from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from app.graph.state import State
from app.graph.nodes import (
    first_tool_call, model_get_schema, query_gen_node, model_check_query,
    format_answer, should_continue, route_after_check, after_answer,
    create_tool_node_with_fallback, get_sql_tools
)
from app.graph.nlg import narrate_answer 
from app.graph.schema_facts import inject_schema_facts
from app.core.tools import db_query_tool

_app_singleton = None


def build_graph():
    list_tables_tool, get_schema_tool = get_sql_tools()

    workflow = StateGraph(State)
    workflow.add_node("first_tool_call", first_tool_call)
    workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
    workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
    workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
    workflow.add_node("inject_schema", inject_schema_facts)
    workflow.add_node("model_get_schema", model_get_schema)
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("correct_query", model_check_query)
    workflow.add_node("format_answer", format_answer)
    workflow.add_node("narrate_answer", narrate_answer)

    workflow.add_edge(START, "first_tool_call")
    workflow.add_edge("first_tool_call", "list_tables_tool")
    workflow.add_edge("list_tables_tool", "model_get_schema")
    workflow.add_edge("model_get_schema", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "inject_schema")
    workflow.add_edge("inject_schema", "query_gen")
    workflow.add_conditional_edges("query_gen", should_continue)
    workflow.add_conditional_edges("correct_query", route_after_check)
    workflow.add_edge("execute_query", "format_answer")
    workflow.add_conditional_edges("format_answer", after_answer, {
        "query_gen": "query_gen",
        "narrate_answer": "narrate_answer",
        "__end__": END,
    })

    app = workflow.compile(checkpointer=MemorySaver())
    return app


def get_graph_app():
    global _app_singleton
    if _app_singleton is None:
        _app_singleton = build_graph()
    return _app_singleton

# Public runner used by API
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from app.utils.messages import random_uuid, invoke_graph


def run_graph(message: str, recursive_limit: int = 30) -> dict:
    app = get_graph_app()
    config = RunnableConfig(recursion_limit=recursive_limit, configurable={"thread_id": random_uuid()})
    inputs = {"messages": [HumanMessage(content=message)]}
    invoke_graph(app, inputs, config)
    return app.get_state(config).values
