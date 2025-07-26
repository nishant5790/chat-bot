import streamlit as st
from typing import TypedDict, Optional, List , Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

# Define the type for the agent state
def get_agent():
    class AgentState(TypedDict):
        messages: List[Union[HumanMessage, AIMessage]]

    llm = ChatOpenAI(model="gpt-4o-mini")

    def process(state):
        response = llm.invoke(state['messages'])
        state['messages'].append(AIMessage(content=response.content))
        return state

    graph = StateGraph(AgentState)
    graph.add_node("process_node", process)
    graph.add_edge(START, "process_node")
    graph.add_edge("process_node", END)
    return graph.compile()

agent = get_agent()

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

st.title("Chatbot App")

st.write("Type your message below and interact with the chatbot.")

user_input = st.text_input("Your message", "")

if st.button("Send") and user_input:
    st.session_state.conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": st.session_state.conversation_history})
    st.session_state.conversation_history = result['messages']

if st.session_state.conversation_history:
    st.subheader("Conversation History")
    for message in st.session_state.conversation_history:
        if isinstance(message, HumanMessage):
            st.markdown(f"**User:** {message.content}")
        elif isinstance(message, AIMessage):
            st.markdown(f"**AI:** {message.content}")

if st.button("Save Conversation"):
    with open("conversation_history.txt", "w") as f:
        f.write("Conversation History:\n")
        for message in st.session_state.conversation_history:
            if isinstance(message, HumanMessage):
                f.write(f"User: {message.content}\n")
            elif isinstance(message, AIMessage):
                f.write(f"AI: {message.content}\n")
    st.success("Conversation saved to conversation_history.txt")
