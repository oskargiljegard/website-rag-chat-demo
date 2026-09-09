#!/usr/bin/env -S uv run --env-file .env

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-5.6-luna"
)

def llm_call(state: MessagesState):
    response = llm.invoke(state["messages"])
    return MessagesState(messages=[response])

def main():
    print("Hello from backend!")
    graph = StateGraph(MessagesState)
    graph.add_node(llm_call)
    graph.add_edge(START, "llm_call")
    graph.add_edge("llm_call", END)
    graph = graph.compile()

    print(graph.invoke(MessagesState(messages=[HumanMessage("hi")])))


if __name__ == "__main__":
    main()

