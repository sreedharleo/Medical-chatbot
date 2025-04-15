 
from flask import Flask, render_template, jsonify, request 
from src.helper import download_hugging_face_embeddings 
from langchain_pinecone import PineconeVectorStore 
from langchain_openai import OpenAI 
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents 
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

#Embed each chunk and upsert the embeddings into your Pinecone index. 
import langchain_pinecone
from langchain_pinecone import PineconeVectorStore 

docsearch = PineconeVectorStore.from_documents( 
    documents=text_chunks, 
    index_name= "medicalbot", 
    embedding=embeddings,
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages( 
    [
        ("system", system_prompt), 
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt) 
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
