import streamlit as st
import requests 

# FastAPI backend URL
api_url = "http://127.0.0.1:8000/ask"

st.title("CBA Credit Card FAQ Chatbot")
st.write("Ask a question and get an AI-powered response!")

# User input 
query = st.text_input("Enter your question")

if st.button("Ask"):
    if query:
        response = requests.post(api_url, json={"query": query})
        
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
            st.write(f"**Answer:** {answer}")
        else:
            st.write("Error: Could not get a response from the backend.")
    else:
        st.write("Please enter a question.")