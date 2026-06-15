# to run using pytest: pytest . -s -v



from graph.chains.retrieval_grader import DocumentGrader, retrieval_grader_chain
from ingestion import retriever


def test_foo():
    assert 1 == 1


def test_retrieval_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: DocumentGrader = retrieval_grader_chain.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == 'yes'
