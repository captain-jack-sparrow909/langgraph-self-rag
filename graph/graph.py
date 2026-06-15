# here we build the graph:
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.chains.answer_grader import answer_grader_chain
from graph.chains.hallucination_grader import hallucination_grader_chain
from graph.const import RETRIEVE, GENERATE, GRADE_DOCUMENT, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


# function for nodes:
def grade_generation_grounded_in_documents_question(state: GraphState)->str:
    print("--CHECK HALLUCINATION--")
    question = state['question']
    documents = state['documents']
    generation = state['generation']

    score = hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
    if hallucination_grade := score.binary_score:
        print("--GENERATION IS GROUNDED IN DOCUMENTS--")
        print("--GRADE GENERATION vs QUESION--")
        answer_score = answer_grader_chain.invoke({"question": question, "generation": generation})
        if answer_grade := answer_score.binary_score:
            print("--DECISION: GENERATION ADDRESSES QUESTION--")
            return "useful"
        else:
            print("--DECISION: GENERATION DOESN'T ADDRESS THE QUESTION")
            return "not useful"
    else:
        print("--DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRY--")
        return "not supported"

# := is called Walrus operator, means assign and use at the same time
# # Without walrus (verbose)
# n = len(data)
# if n > 10:
#     print(n)

# # With walrus (concise)
# if (n := len(data)) > 10:
#     print(n)


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

workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_question,     {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEB_SEARCH,
    })

workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
# app.get_graph().draw_mermaid_png(output_file_path="mermaid.png")

