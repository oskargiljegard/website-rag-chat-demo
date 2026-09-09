from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

def main():
    print("Hello from backend!")
    graph = StateGraph(MessagesState)
    graph.add_node(mock_llm)
    graph.add_edge(START, "mock_llm")
    graph.add_edge("mock_llm", END)
    graph = graph.compile()

    print(graph.invoke(MessagesState(messages=[HumanMessage("hi")])))


if __name__ == "__main__":
    main()

