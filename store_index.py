from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



extracted_data=load_pdf_file(data='C:\\Users\\Sreedhar R\\Medical-chatbot\\Data\\Current Essentials of Medicine(1)(1).pdf')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()



from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec
import os

pc = Pinecone(api_key=PINECONE_API_KEY)


index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


docsearch = PineconeVectorStore.from_documents( 
    documents=text_chunks, 
    index_name= "medicalbot", 
    embedding=embeddings,
)