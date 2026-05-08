import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document # type: ignore
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA

import requests
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.embeddings import HuggingFaceEmbeddings
import os

persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
model = os.environ.get("MODEL", "llama3")

# Define a function to scrape web content
# def scrape_web_content(urls):
#     docs = []
#     for url in urls:
#         try:
#             response = requests.get(url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.text, 'html.parser')
#             text = soup.get_text()
#             docs.append(Document(page_content=text, metadata={"url": url}))
#         except requests.RequestException as e:
#             st.error(f"Error fetching {url}: {e}")
#     return docs

# Define the chatbot processing function
def process_input(question, chat_history):
        # embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    model_local = Ollama(model=model)  # Update the port if needed
    
    # Define the RAG prompt template for the chatbot
    rag_prompt_template = """
    You are a customer service assistant for a supermarket. Answer the customer's question based on the information you have.

    {chat_history}
    Relevant Information: {context}
    User: {question}
    Assistant:
    """
    embedding=embeddings.OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    
    # Web scraping URLs
    # urls = ["https://www.jiomart.com/"]  # Replace with actual URLs
    # docs = scrape_web_content(urls)
    # print(docs)
    
    # Split the text into chunks
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    # doc_splits = text_splitter.split_documents(docs)
    
    # # Convert text chunks into embeddings and store in vector database
    # vectorstore = Chroma.from_documents(
    #     documents=doc_splits,
    #     collection_name="rag-chroma",
    #     embedding=embeddings.OllamaEmbeddings(model="nomic-embed-text"),
    # )
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Retrieve relevant information
    relevant_info = qa(question)
    
    # Combine chat history and relevant information with the new question# Assuming each 'info' in 'relevant_info' is a Document object
    # context = "\n".join([info.page_content for info in relevant_info])
    print(relevant_info)

    # Combine chat history and relevant information with the new question
    chat_history_str = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in chat_history])
    prompt = ChatPromptTemplate.from_template(rag_prompt_template.format(chat_history=chat_history_str, context=relevant_info, question=question))

    
    input_chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | model_local
        | StrOutputParser()
    )
    return relevant_info['result']

# Streamlit App
st.title("Supermarket Customer Service Chatbot")
st.write("Ask any questions related to the supermarket, and the chatbot will assist you.")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Handle the "New Chat" button
if st.button('New Chat'):
    st.session_state.chat_history = []

# Input field for the question
question = st.text_input("Your question")

# Button to process input
if st.button('Ask'):
    if question:
        with st.spinner('Processing...'):
            answer = process_input(question, st.session_state.chat_history)
            st.session_state.chat_history.append({"user": question, "assistant": answer})
            st.text_area("Answer", value=answer, height=300, disabled=True)

# Display the chat history
if st.session_state.chat_history:
    st.write("Chat History:")
    for i, entry in enumerate(st.session_state.chat_history):
        st.text_area(f"You ({i+1})", value=entry["user"], height=50, disabled=True)
        st.text_area(f"Assistant ({i+1})", value=entry["assistant"], height=100, disabled=True)