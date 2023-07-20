from dotenv import load_dotenv

load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


pdf_path = "./stack_care_white_paper.pdf"
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
split_docts = splitter.split_documents(documents=documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=split_docts, embedding=embeddings)
vectorstore.save_local("faiss_index_stackcare")

restored_vectorstore = FAISS.load_local("faiss_index_stackcare", embeddings)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=restored_vectorstore.as_retriever()
)

result = qa.run("What does StackCare detect?")
print(result)

result = qa.run("Describe a real use case example of StackCare")
print(result)
