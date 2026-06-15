# this chain is going to receive question and retrieved documents, and is going to determine whether the retrieved document is 
# relevant to the question or not.

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")

# defining schema to force the LLM for a certain output
class DocumentGrader(BaseModel):
    """
    Binary check for relevance check on retrieved documents.
    """
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_with_grader = llm.with_structured_output(DocumentGrader)
# behind the scene with_structured_output is using the same mechanism for getting structured ouput.
# as in : llm = llm.bind_tools( tools=[AnswerQuestion], tool_choice="AnswerQuestion") but without having to call a tool

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n{document} \n\n User question: {question}")
])

retrieval_grader_chain = grade_prompt_template | structured_llm_with_grader

