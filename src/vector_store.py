import os
import uuid
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings

# Import custom modules
from src.data_loader import load_data
from src.text_splitter import split_text


# Set up API key
load_dotenv()
OPENAI_API_KEY =  os.environ.get("OPENAI_API_KEY")


def generate_id(text):
    # Generate unique chunk id based on content 
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text)) 

def create_vector_db(chunked_data, persist_directory="./chroma_db/cc_faq_openai", collection_name="cc_faq"):
    # Prepare chunked data for embedding
    ids = [generate_id(chunk["content"]) for chunk in chunked_data] # Generate unique UUID based on content
    texts = [chunk["content"] for chunk in chunked_data]
    metadatas = [{"url": chunk["url"], "title": chunk["title"], "segment": chunk["segment"]} for chunk in chunked_data]

    # Create a Chroma vector store
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key = OPENAI_API_KEY
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # Check existing IDs
    existing = vectorstore.get()
    existing_ids = set(existing["ids"]) if existing and "ids" in existing else set() #Get stored IDs

    new_texts, new_ids, new_metadatas = [], [], []
    for i, chunk_id in enumerate(ids):
        if chunk_id not in existing_ids:
            new_texts.append(texts[i])
            new_ids.append(chunk_id)
            new_metadatas.append(metadatas[i])
    
    # Add only new chunks 
    if new_texts:
        vectorstore.add_texts(ids=new_ids, texts=new_texts, metadatas=new_metadatas)
        print(f"Added {len(new_texts)} new chunks.")
    else: 
        print("No new chunks added.")
    
    return vectorstore

    
if __name__ == "__main__":
    # load and chunk data
    scraped_data = load_data("/Users/alexis/Desktop/Learning/Projects/202502_Custom_Chatbot/data/scraped_data_w_segment.json")
    chunked_data = split_text(scraped_data)

    # create the vector database and get the collection
    vectorstore = create_vector_db(chunked_data)
    print(f"There are {vectorstore._collection.count()} chunks in vectorstore.")
