from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")

class GradeHallucination(BaseModel):
    """Binary score for hallucination present in the generated answer"""
    binary_score: bool = Field(description="Answer is grounded in the fact, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "set of facts: \n\n {documents} LLM generation: {generation}")
])

hallucination_grader_chain: RunnableSequence = hallucination_prompt | structured_llm_grader

