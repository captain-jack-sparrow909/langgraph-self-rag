# here we build the graph:
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.const import RETRIEVE, GENERATE, GRADE_DOCUMENT, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


# function for nodes:
def decide_to_generate(state):
    if state['web_search']:
        return WEB_SEARCH
    else:
        return GENERATE


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENT, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENT)

workflow.add_conditional_edges(GRADE_DOCUMENT, decide_to_generate, {
    WEB_SEARCH: WEB_SEARCH, 
    GENERATE: GENERATE
})

workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
# app.get_graph().draw_mermaid_png(output_file_path="mermaid.png")

