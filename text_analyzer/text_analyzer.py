import os
from dotenv import load_dotenv

load_dotenv()

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI

import pinecone

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
)


if __name__ == "__main__":
    loader = TextLoader("./medium_article.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_text = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    doc_search = Pinecone.from_documents(
        split_text, embeddings, index_name=os.environ.get("PINECONE_INDEX_NAME")
    )

    # deprecated, should use RetrievalQA
    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        vectorstore=doc_search,
        return_source_documents=False,
    )

    query = "What features should a vector database have? Only list the features, without explaining anything else."
    result = qa({"query": query})
    print(result)
