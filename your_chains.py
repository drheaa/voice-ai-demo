from langchain_openai import ChatOpenAI
from langchain.tools import tool

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
