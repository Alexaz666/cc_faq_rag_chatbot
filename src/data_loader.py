import json

def load_data (json_path):
    """
    Load scraped data in the JSON file from the specified file path

    Args: 
        json_path (str): Path to the JSON file.

    Returns:
        list[dict]: List of scraped data entries.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        scraped_data = json.load(file)
        
    print(f"Loaded {len(scraped_data)} entries from {json_path}")
    return scraped_data
    


# Test
if __name__ == "__main__":
    sample_data = load_data("/Users/alexis/Desktop/Learning/Projects/202502_Custom_Chatbot/data/scraped_data_w_segment.json")
    print("Sample:", sample_data[:1])  