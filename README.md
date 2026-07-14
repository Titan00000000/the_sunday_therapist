# The Sunday Therapist RAG based chatbot

An intelligent, Retrieval-Augmented Generation (RAG) chatbot designed to act as a knowledgeable assistant for the Sunday Therapist. This system utilises a tiered hybrid retrieval architecture built using Python, Numpy, and the Gemini API, wrapped in a user-friendly Streamlit interface.

## Project Overview
This project automates the end-to-end process of:
* **Web Scraping:** Extracting and cleaning text data from 5 public pages of the website https://www.thesundaytherapist.co.uk.
* **Synthetic Q/A Generation:** Leveraging an LLM to generate a high-quality dataset of 200 domain-specific question-and-answer pairs.
* **Single-Index Vector Store:** Vectorises the FAQ dataset using Gemini Embeddings and performs lightning-fast semantic searches using NumPy-based cosine similarity in-memory.
* **Hybrid Routing Logic:** Employing a distance-based fallback routing system (threshold < 0.75) to deliver highly accurate answers.

---

## Tech Stack & Frameworks
* **Language:** Python 
* **Frontend UI:** Streamlit
* **Vector Database:** NumPy for easy storage online as this is a small website
* **Embedding & LLM Generation:** Google Gemini API 
* **Web Scraping:** Beautiful Soup / Requests

---

## Author
* **Mr Timur Rahman**