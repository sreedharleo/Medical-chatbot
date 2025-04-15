from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings



#Extract Data from the pdf file


def load_pdf_file(data):
    loader = PyPDFLoader(data)
    documents = loader.load()
    return documents   



#Split the Data into Text Chunks 

def text_split(extracted_data): 
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20) 
    text_chunks=text_splitter.split_documents (extracted_data) 
    return text_chunks


#Download the Embeddings from Hugging Face 

def download_hugging_face_embeddings(): 
    embeddings= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 
    return embeddings  