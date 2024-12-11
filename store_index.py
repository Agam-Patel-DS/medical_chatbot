from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os


load_dotenv()

PINECONE_API_KEY=os.getenv('PINECONE_TOKEN')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data=load_pdf_file(data='data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


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

import pinecone
from langchain.vectorstores import Pinecone

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = Pinecone.from_documents(
    documents=text_chunks,  # Replace `chunks` with your list of documents
    embedding=embeddings,
    index_name=index_name
)
