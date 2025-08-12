import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
# --------------------

load_dotenv()
# STREAMLIT APP CONFIG
# --------------------
st.set_page_config(page_title="YouTube Transcript Chatbot", page_icon="🎥", layout="wide")
st.title("🎥 YouTube Transcript Chatbot")
st.write("Ask questions about a YouTube video's transcript.")

# --------------------
# API KEYS (SET YOURS HERE OR IN .env)
# --------------------
groq_api_key = os.getenv("GROQ_API_KEY")
huggingface_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"] = huggingface_token
# --------------------
# VIDEO ID INPUT
# --------------------
video_id = st.text_input("Enter YouTube Video ID (e.g., nABrf7oM7A0):", value="")

if st.button("Load Transcript"):
    with st.spinner("Fetching transcript..."):
        try:
            api_instance = YouTubeTranscriptApi()
            transcript_list = api_instance.fetch(video_id=video_id, languages=['en','hi'])
            transcript_list = transcript_list.to_raw_data()
            transcript = " ".join([chunk["text"] for chunk in transcript_list])

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.create_documents([transcript])

            # Create embeddings & FAISS store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(docs, embeddings)

            st.session_state.vector_store = vector_store
            st.success("Transcript loaded and indexed successfully!")

        except TranscriptsDisabled:
            st.error("No captions available for this video.")
        except Exception as e:
            st.error(f"Error: {e}")

# --------------------
# CHAT SECTION
# --------------------
if "vector_store" in st.session_state:
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    st.subheader("💬 Chat with the Transcript")
    user_input = st.text_input("Ask a question about the video:")

    if st.button("Get Answer") and user_input.strip():
        with st.spinner("Generating answer..."):
            answer = main_chain.invoke(user_input)
            st.markdown(f"**Answer:** {answer}")

else:
    st.info("Please load a transcript first before asking questions.")
