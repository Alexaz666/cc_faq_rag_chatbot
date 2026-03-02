"""
This module handles text chunking for a Retrieval-Augmented Generation (RAG) pipeline.
It splits long scraped texts into smaller overlapping chunks using LangChain's
RecursiveCharacterTextSplitter, and removes duplicates to ensure efficient vector storage.
"""


from src.data_loader import load_data # import the data_load function
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text (data, chunk_size=500, chunk_overlap=100):
    """
    Splits a list of text entries into overlapping chunks, removing duplicates.

    Args:
        data (list): A list of dictionaries with keys 'url', 'title', and 'content'.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.

    Returns:
        list: A list of dictionaries, each representing a unique text chunk with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunked_data = [] # list to store the resulting chunks
    seen_chunks = set() # set to keep track of unique chunks

    for entry in data:
        # In scraped_data.json 'content' contains actual scraped text
        chunks = splitter.split_text(entry["content"])

        for chunk in chunks:
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                chunked_data.append({
                    "url": entry["url"],
                    "title": entry["title"],
                    "segment": entry["segment"],
                    "content": chunk 
                })
    return chunked_data


# Test
if __name__ == "__main__":
    # load data
    scraped_data = load_data("/Users/alexis/Desktop/Learning/Projects/202502_Custom_Chatbot/data/scraped_data_w_segment.json")
    
    # chunking data
    chunked_data = split_text(scraped_data)

    print(f"Retured {len(chunked_data)} chunks from {len(scraped_data)} entries.")
    print("sample chunk:", chunked_data[0])
