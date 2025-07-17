from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, List

# Import your LangChain LLM + tools chain
from your_chains import llm_with_tools  # Assume this is defined elsewhere

# Define state with a message history
class MessagesState(TypedDict):
    messages: List
    last_intent: str

# Define reusable function to invoke the LLM with a system prompt
def llm_node(system_prompt: str):
    def node(state: MessagesState) -> MessagesState:
        sys_msg = SystemMessage(content=system_prompt)
        updated_messages = llm_with_tools.invoke([sys_msg] + state["messages"])
        state["messages"] = updated_messages
        return state
    return node

# Define system prompts for each node
prompts = {
    "Start": "You are a Voice AI assistant for a pizza restaurant demo. Your job is to help users place delivery or pickup orders, make table reservations or handle complaints. Greet the user and wait for their input. Keep the conversation friendly, concise and helpful.",
    "SpeechToText": "Use ElevenLabs STT to transcribe incoming audio from the user. Ensure accuracy and pass the clean transcription to the Intent Classifier.",
    "IntentClassifier": "Classify the user's intent based on the transcription. The possible intents are: order, reservation, complaint. Output one of these three categories and route to the corresponding flow.",
    "OrderFlow": "Initiate an order flow. Ask the user whether they want pickup or delivery, and guide them through selecting items from the menu.",
    "CheckMissingOrderInfo": "Check the order for missing information(User Name, Phone Number, Delivery Address, Items Selected, Quantity or Special Requests) and ask only the missing parts.",
    "OrderConfirmation": "Summarize the full order. Repeat items, pickup/delivery details, and price. End with: “Thanks for placing your order. This is a demo of my capabilities, I hope you enjoyed the experience. How else can I assist you today?”",
    "ReservationFlow": "Initiate a reservation flow. Ask the user for their details and preferences for the booking.",
    "GatherReservationInfo": "Ask for name, phone number, date and time, number of people, and table preference.",
    "CheckMissingReservationInfo": "Validate that all reservation details are complete. If any are missing, ask only the missing ones.",
    "ReservationConfirmation": "Confirm the reservation summary. Say: “Thanks! Your reservation is confirmed. I’ve sent you a confirmation SMS. This is a demo of my capabilities. How else can I help you today?”",
    "ComplaintFlow": "Detect complaints or frustration in the user's tone or keywords. If a complaint is detected, route here immediately.",
    "LogComplaint": "Ask: “Would you like me to take a note and let the manager know to call you back?” Log the complaint details to Google Sheet.",
    "LiveHandoff": "If the user insists on speaking with someone immediately, say: “No problem. This is a demo of my capabilities, I will forward your call now.” Trigger a warm handoff and log the full conversation history with caller ID to the complaint sheet.",
    "SaveToSheet": "Store all collected information from the order or reservation into Google Sheet in the appropriate tab. Ensure timestamps, caller ID, and interaction type are saved.",
    "TextToSpeech": "Convert the assistant's response to audio using ElevenLabs TTS and return it to the user in real time.",
    "CheckDone": "Ask the user: “Would you like to do anything else—perhaps place another order or make a reservation?”",
    "LoopBack": "Restart the previous flow if the user wants to continue.",
    "FinalExit": "Say goodbye to the user. “Thanks for calling. Have a great day!” Then end the call/session."
}

# Build the graph
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("Start", llm_node(prompts["Start"]))
graph.add_node("SpeechToText", llm_node(prompts["SpeechToText"]))
graph.add_node("IntentClassifier", llm_node(prompts["IntentClassifier"]))
graph.add_node("OrderFlow", llm_node(prompts["OrderFlow"]))
graph.add_node("CheckMissingOrderInfo", llm_node(prompts["CheckMissingOrderInfo"]))
graph.add_node("OrderConfirmation", llm_node(prompts["OrderConfirmation"]))
graph.add_node("ReservationFlow", llm_node(prompts["ReservationFlow"]))
graph.add_node("GatherReservationInfo", llm_node(prompts["GatherReservationInfo"]))
graph.add_node("CheckMissingReservationInfo", llm_node(prompts["CheckMissingReservationInfo"]))
graph.add_node("ReservationConfirmation", llm_node(prompts["ReservationConfirmation"]))
graph.add_node("ComplaintFlow", llm_node(prompts["ComplaintFlow"]))
graph.add_node("LogComplaint", llm_node(prompts["LogComplaint"]))
graph.add_node("LiveHandoff", llm_node(prompts["LiveHandoff"]))
graph.add_node("SaveToSheet", llm_node(prompts["SaveToSheet"]))
graph.add_node("TextToSpeech", llm_node(prompts["TextToSpeech"]))
graph.add_node("CheckDone", llm_node(prompts["CheckDone"]))
graph.add_node("LoopBack", llm_node(prompts["LoopBack"]))
graph.add_node("FinalExit", llm_node(prompts["FinalExit"]))

# Define transitions
graph.set_entry_point("Start")
graph.add_edge("Start", "SpeechToText")
graph.add_edge("SpeechToText", "IntentClassifier")

# Conditional routing based on intent
intent_router = lambda state: (
    state.update({"last_intent": "order"}) or "OrderFlow" if "order" in state["messages"][-1].content.lower() else
    state.update({"last_intent": "reservation"}) or "ReservationFlow" if "reservation" in state["messages"][-1].content.lower() else
    state.update({"last_intent": "complaint"}) or "ComplaintFlow" if "complaint" in state["messages"][-1].content.lower() else
    "FinalExit"
)
graph.add_conditional_edges("IntentClassifier", intent_router)

# Order flow
graph.add_edge("OrderFlow", "CheckMissingOrderInfo")
graph.add_edge("CheckMissingOrderInfo", "OrderConfirmation")
graph.add_edge("OrderConfirmation", "SaveToSheet")

# Reservation flow
graph.add_edge("ReservationFlow", "GatherReservationInfo")
graph.add_edge("GatherReservationInfo", "CheckMissingReservationInfo")
graph.add_edge("CheckMissingReservationInfo", "ReservationConfirmation")
graph.add_edge("ReservationConfirmation", "SaveToSheet")

# Complaint flow
graph.add_edge("ComplaintFlow", "LogComplaint")
graph.add_edge("LogComplaint", "LiveHandoff")
graph.add_edge("LiveHandoff", "FinalExit")

# Final sequence
graph.add_edge("SaveToSheet", "TextToSpeech")
graph.add_edge("TextToSpeech", "CheckDone")

def loopback_router(state):
    return {
        "order": "OrderFlow",
        "reservation": "ReservationFlow",
        "complaint": "ComplaintFlow"
    }.get(state.get("last_intent", "order"), "OrderFlow")

done_check = lambda state: (
    loopback_router(state) if "yes" in state["messages"][-1].content.lower() else "FinalExit"
)
graph.add_conditional_edges("CheckDone", done_check)

graph.set_finish_point("FinalExit")

# Compile the graph
app = graph.compile()