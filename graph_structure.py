from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from src.state import GraphState
from src.nodes.stt_node import stt_node
from src.nodes.intent_classifier_node import intent_classifier_node
from src.nodes.order_flow_node import order_flow_node
from src.nodes.reservation_flow_node import reservation_flow_node
from src.nodes.complaint_flow_node import complaint_flow_node
from src.nodes.check_missing_info_node import check_missing_info_node
from src.nodes.order_confirmation_node import order_confirmation_node
from src.nodes.reservation_confirmation_node import reservation_confirmation_node
from src.nodes.log_to_sheets_node import log_to_sheets_node
from src.nodes.tts_node import tts_node
from src.nodes.exit_node import exit_node
from src.nodes.loopback_node import loopback_node

# Initialize the graph
workflow = StateGraph(GraphState)

# Register all nodes
workflow.add_node("STT", stt_node)
workflow.add_node("IntentClassifier", intent_classifier_node)
workflow.add_node("OrderFlow", order_flow_node)
workflow.add_node("ReservationFlow", reservation_flow_node)
workflow.add_node("ComplaintFlow", complaint_flow_node)
workflow.add_node("CheckMissingInfo", check_missing_info_node)
workflow.add_node("OrderConfirmation", order_confirmation_node)
workflow.add_node("ReservationConfirmation", reservation_confirmation_node)
workflow.add_node("LogToSheets", log_to_sheets_node)
workflow.add_node("TTS", tts_node)
workflow.add_node("LoopBack", loopback_node)
workflow.add_node("Exit", exit_node)

# Define entry point
workflow.set_entry_point("STT")

# Define transitions
workflow.add_edge("STT", "IntentClassifier")

workflow.add_conditional_edges("IntentClassifier", {
    "order": "OrderFlow",
    "reservation": "ReservationFlow",
    "complaint": "ComplaintFlow",
    "unknown": "Exit"
})

# Order flow
workflow.add_edge("OrderFlow", "CheckMissingInfo")
workflow.add_edge("CheckMissingInfo", "OrderConfirmation")
workflow.add_edge("OrderConfirmation", "LogToSheets")
workflow.add_edge("LogToSheets", "TTS")
workflow.add_edge("TTS", "LoopBack")
workflow.add_edge("LoopBack", "IntentClassifier")

# Reservation flow
workflow.add_edge("ReservationFlow", "CheckMissingInfo")
workflow.add_edge("CheckMissingInfo", "ReservationConfirmation")
workflow.add_edge("ReservationConfirmation", "LogToSheets")
workflow.add_edge("LogToSheets", "TTS")
workflow.add_edge("TTS", "LoopBack")
workflow.add_edge("LoopBack", "IntentClassifier")

# Complaint flow
workflow.add_edge("ComplaintFlow", "LogToSheets")
workflow.add_edge("LogToSheets", "TTS")
workflow.add_edge("TTS", "LoopBack")
workflow.add_edge("LoopBack", "IntentClassifier")

# Exit
workflow.add_edge("Exit", "TTS")

# Compile the app
app = workflow.compile()

