from dotenv import load_dotenv

load_dotenv()

from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pNcQ5XXMgH4&ab_channel=GregKamradt%28DataIndy%29",
    add_video_info=True,
)

video_result = loader.load()
print(
    f"Found video from {video_result[0].metadata['author']} with a duration of {video_result[0].metadata['length']} seconds."
)

model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
llm = ChatOpenAI(temperature=0, model_name=model_name)

# for short videos, the transcript can be directly used
summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff", verbose=False)
summary = summarize_chain.run(video_result)
print(summary)

# for long videos, the transcript must be split into multiple documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
video_texts = text_splitter.split_documents(video_result)

# map_reduce generates a summary of each document and then generates a summary of the summaries
summarize_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)
summary = summarize_chain.run(video_result)
print(summary)
