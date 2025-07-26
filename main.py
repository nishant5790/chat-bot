from typing import TypedDict, Optional, List , Union
from langchain_core.messages import HumanMessage , AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from print_color import print
from dotenv import load_dotenv
load_dotenv()

# Define the type for the agent state
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Initialize the LLM model
llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    """
    Process the agent state by appending a new message.
    """
    # Append a new message to the state
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    
    print(f"{response.content}\n", tag="AI:",tag_color="green",color="yellow")
    return state

# Define the graph with a single node that processes the state
graph = StateGraph(AgentState)

graph.add_node("process_node", process)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter your message: ")
while user_input.lower() != "exit" :
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})

    conversation_history = result['messages']
    user_input = input("Enter your message (or type 'exit' to quit): ")

with open("conversation_history.txt", "w") as f:
    f.write("Conversation History:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
print("Exiting the chatbot.")
