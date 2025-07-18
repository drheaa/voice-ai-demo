from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage

# Define mock tools to be used in the LangGraph demo

@tool
def save_to_sheets(data: dict) -> str:
    """Mock tool: Save order or reservation data to Google Sheets."""
    print("[Tool] Saving to Google Sheets:", data)
    return "Saved to Google Sheets."

@tool
def get_menu(_: dict = {}) -> dict:
    """Mock tool: Return the current menu as a dictionary."""
    return {
        "pizzas": [
            {"name": "Margherita", "price": 12, "tags": ["vegetarian"]},
            {"name": "Pepperoni", "price": 14},
            {"name": "BBQ Chicken", "price": 15},
        ],
        "sides": [
            {"name": "Garlic Bread", "price": 5},
            {"name": "Chicken Wings", "price": 8}
        ],
        "beverages": [
            {"name": "Coke", "price": 3},
            {"name": "Water", "price": 2}
        ]
    }

@tool
def send_sms(data: dict) -> str:
    """Mock tool: Send a confirmation SMS to the user."""
    print("[Tool] Sending SMS:", data)
    return "SMS sent successfully."

# Instantiate LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind tools to LLM
llm_with_tools = llm.bind_tools([save_to_sheets, get_menu, send_sms])

# Tool call handler function
def handle_tool_calls(response, state, llm_with_tools):
    if hasattr(response, "tool_calls"):
        for tool_call in response.tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id

            tool_fn = llm_with_tools.tools.get(tool_name)

            if tool_fn:
                tool_output = tool_fn.invoke(tool_args)
                state["messages"].append(
                    ToolMessage(tool_call_id=tool_id, content=tool_output)
                )
