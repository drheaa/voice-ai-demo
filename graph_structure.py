
from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
import json

with open("menu.md", "r") as f:
    menu_md = f.read()

class MessagesState(TypedDict):
    messages: List

system_prompt = f"""

Start the conversation with: "Welcome to Bellini’s. I am here to help you understand what’s on our menu. What would you like to know?"

You must ONLY answer questions related to the menu provided below.

Here is the menu:

{menu_md}

Whenever a user asks about a specific menu item (or multiple items), extract the item(s) in a JSON format like this:

{{
  "mentioned_items": [
    {{
      "name": "Vegorama",
      "category": "Pizza"
    }},
    ...
  ]
}}

Respond normally to the user while also including the extracted JSON for logging.

If the user asks something else that is not present in the prompt or the menu.md, then say:
"I'm sorry, I cannot help with this at the moment."

Ask "Do you need anything else?" after each answer.

If the user says "yes": answer the questions related to the menu 
{menu_md}
again.

If the user says: "no": answer "Thanks for calling Bellini's. Your order has been logged and you will receive a SMS confirmation with a payment link."
"""

llm = ChatOpenAI(model="gpt-4o")

def menu_node(state: MessagesState) -> MessagesState:
    msgs = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(msgs)
    state["messages"].append(response)
    return state

graph = StateGraph(MessagesState)

graph.add_node("MenuAgent", menu_node)

graph.set_entry_point("MenuAgent")
graph.set_finish_point("MenuAgent")

app = graph.compile()
>>>>>>> de3bc06 (Basic Menu Agent answering the questions)
