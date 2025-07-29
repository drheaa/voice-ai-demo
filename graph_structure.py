from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
import json

with open("menu.md", "r") as f:
    menu_md = f.read()

class MessagesState(TypedDict):
    messages: List
    last_intent: str

llm = ChatOpenAI(model="gpt-4o")

def make_node(prompt: str):
    def node(state: MessagesState) -> MessagesState:
        msgs = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(msgs)
        state["messages"].append(response)
        return state
    return node

start_prompt = """
You are a friendly and efficient voice assistant for Bellini’s restaurant. Greet the user warmly and ask how you can help with menu questions, orders, reservations, or complaints.

Begin the conversation with this:
"Welcome to Bellini’s. I’m here to help you understand what’s on our menu. What would you like to know?"

Only assist with menu-related requests. If the user says something unrelated, politely say:
"I'm sorry, I can only help with Bellini’s menu-related questions at the moment."
"""

intent_classifier_prompt = """
Based on the user message, classify their intent into one of these categories:
- order
- reservation
- complaint
- menu_question

Look at their tone and content. Respond in a friendly way, but internally extract the correct intent and route accordingly.
"""

order_flow_prompt = f"""
Begin the order flow.

Ask the user:
- "Would you like pickup or delivery?"
- "What item would you like to order?"

Refer to the Bellini’s menu below when asking about items. If unclear, ask follow-up questions politely.

Here is the menu:
{menu_md}
"""

menu_agent_prompt = f"""
Use the menu below to help answer specific product questions. Respond in a helpful tone and include a JSON of mentioned items for tracking.

Menu:
{menu_md}

Format JSON like:
{{
  "mentioned_items": [
    {{
      "name": "Vegorama",
      "category": "Pizza"
    }},
    ...
  ]
}}

Always end by asking: “Do you need anything else?”

If yes: continue answering.

If no: say “Thanks for calling Bellini’s. Your order has been logged and you will receive a confirmation SMS with a payment link.”
"""

check_missing_info_prompt = """
Check if the order contains all required information: name, phone number, address (if delivery), items, quantity.

If anything is missing, politely ask the user. If the pizza name is invalid, refer them to menu options. Ensure clarity and accuracy in details.
"""

check_done_prompt = """
Ask the user: “Would you like to do anything else—perhaps ask about another menu item or place another order?”

Expect a yes or no answer and route accordingly.
"""

final_exit_prompt = """
Say a warm goodbye to the user:

"Thanks for calling Bellini’s. Have a great day!"

Then end the session.
"""

def intent_router(state: MessagesState) -> str:
    msg = state["messages"][-1].content.lower()
    if "order" in msg:
        state["last_intent"] = "order"
        return "OrderFlow"
    elif "reservation" in msg:
        state["last_intent"] = "reservation"
        return "MenuAgent"  # placeholder for future reservation node
    elif "complaint" in msg:
        state["last_intent"] = "complaint"
        return "MenuAgent"  # placeholder for future complaint node
    else:
        state["last_intent"] = "menu_question"
        return "MenuAgent"

def done_check(state: MessagesState) -> str:
    msg = state["messages"][-1].content.lower()
    if "yes" in msg:
        return "MenuAgent"
    else:
        return "FinalExit"

graph = StateGraph(MessagesState)

graph.add_node("Start", make_node(start_prompt))
graph.add_node("IntentClassifier", make_node(intent_classifier_prompt))
graph.add_node("OrderFlow", make_node(order_flow_prompt))
graph.add_node("MenuAgent", make_node(menu_agent_prompt))
graph.add_node("CheckMissingOrderInfo", make_node(check_missing_info_prompt))
graph.add_node("CheckDone", make_node(check_done_prompt))
graph.add_node("FinalExit", make_node(final_exit_prompt))

graph.set_entry_point("Start")

graph.add_edge("Start", "IntentClassifier")

graph.add_conditional_edges(
    "IntentClassifier",
    intent_router,
    {
        "OrderFlow": "OrderFlow",
        "MenuAgent": "MenuAgent",
    }
)

graph.add_edge("OrderFlow", "MenuAgent")
graph.add_edge("MenuAgent", "CheckMissingOrderInfo")
graph.add_edge("CheckMissingOrderInfo", "CheckDone")

graph.add_conditional_edges(
    "CheckDone",
    done_check,
    {
        "MenuAgent": "MenuAgent",
        "FinalExit": "FinalExit"
    }
)

graph.set_finish_point("FinalExit")

app = graph.compile()
