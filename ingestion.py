from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# URLs to load data from:
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# fetching the content of these urls:
docs = [WebBaseLoader(url).load() for url in urls]
# flatten the docs array:
docs_list = [item for sublist in docs for item in sublist]


# splitting the content into chunks:
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(  # Still splits recursively like above, BUT with token awareness
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)


# storing into chroma db: first part is needed only once
# vector_store = Chroma.from_documents(
#     documents=doc_splits,
#     embedding=OpenAIEmbeddings(),
#     collection_name="chroma-rag",
#     persist_directory="./.chroma"
# )

retriever = Chroma(
    embedding_function=OpenAIEmbeddings(),
    collection_name="chroma-rag",
    persist_directory="./.chroma"
).as_retriever()
