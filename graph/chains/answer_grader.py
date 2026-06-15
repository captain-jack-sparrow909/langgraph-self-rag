from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")

class GradeAnswer(BaseModel):
    binary_score: bool = Field(description="Answer addresses the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "User question: \n\n{question}\n\n LLM generation: {generation}")
])

answer_grader_chain: RunnableSequence = prompt | structured_llm_grader


