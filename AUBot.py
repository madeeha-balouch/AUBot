import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "Add your key here"
st.header("AUBot")
st.subheader('Ask any Ajman University related queries')
st.divider()

#step 1
#Loading unstructured data
loader = PyPDFLoader("AUBot/Student_Handbook_2022-20231.pdf")
pages = loader.load_and_split()

#step 2
#splitting the loaded document into chucks of data 
text_splitter = CharacterTextSplitter(chunk_size = 3000, chunk_overlap = 40)
chunks = text_splitter.split_documents(pages)

#step 3, Adding the embedding model
embeddings_model = OpenAIEmbeddings()

#step 4, Vector stores
db = FAISS.from_documents(chunks, embeddings_model)


question= st.text_input('Type in your query', placeholder="When was Ajman University Founded")
if question:
 docs = db.similarity_search(question)

#step 5, Adding a retreiver
 retriever = db.as_retriever()

#step 6, adding a LLM chain 
 qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), chain_type="stuff",retriever=retriever)
 answer = qa_chain.run(question)
 st.text_area("Output", answer)


