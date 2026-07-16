import csv
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="The Sunday Therapist Assistant", page_icon="💬")

# --- CUSTOM CSS FOR BRANDING, TYPOGRAPHY & PRECISE ALIGNMENT ---
st.markdown("""
    <style>
    /* 1. Force hide the default large Streamlit headers */
    h1, [data-testid="stHeader"] {
        display: none !important;
    }

    /* 2. Remove all massive top margin/padding space from the main block */
    .block-container {
        padding-top: 10px !important;
        padding-bottom: 95px !important; /* Space for fixed bottom elements */
    }
    
    /* 3. Global font styling - Force uniform font sizes across the entire app */
    html, body, [class*="css"], .stMarkdown, p, span, div, caption, .stTextInput, textarea, input, label {
        font-size: 15px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Target the placeholder text specifically inside the input bar */
    input::placeholder {
        font-size: 15px !important;
    }
    
    /* 4. Custom Title styled exactly like body text, bolded and underlined */
    .therapy-title {
        font-size: 15px !important;
        font-weight: bold !important;
        text-decoration: underline !important;
        margin-top: 0px !important;
        margin-bottom: 5px !important;
        color: #31333F !important;
    }

    /* 5. Subtitle/Caption matching the body size */
    .therapy-caption {
        font-size: 15px !important;
        color: #555555 !important;
        margin-bottom: 15px !important;
    }
    
    /* 6. Shift the input field container to the right to make room on the left */
    [data-testid="stChatInput"] {
        margin-left: 95px !important;
    }
    
    /* 7. Position container to sit alongside the input field at the bottom */
    .custom-close-container {
        position: fixed;
        bottom: 24px; /* Perfectly aligns with Streamlit's chat input bar container */
        left: 15px;
        z-index: 999999;
    }
    
    /* 8. Make the close button perfectly match the height & structure of the chat input */
    .my-close-btn {
        background-color: #3A574B !important; /* Forest green */
        color: white !important;
        border-radius: 8px !important;       /* Clean match to the prompt container */
        border: 1px solid transparent !important;
        height: 40px !important;              /* Matches Streamlit's text input inner height */
        padding: 0px 14px !important;
        font-size: 15px !important;
        font-weight: bold !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        transition: background-color 0.2s ease;
    }
    
    .my-close-btn:hover {
        background-color: #2e443b !important;
    }
    </style>
    
    <div class="custom-close-container">
        <button class="my-close-btn" onclick="window.parent.postMessage('minimize_chat', '*')">✕ Close</button>
    </div>
""", unsafe_allow_html=True)

# --- BRANDED TITLE & SUBTEXT ---
st.markdown('<p class="therapy-title">The Sunday Therapist Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="therapy-caption">Hello and welcome to the automated assistance chatbot for treatments, pricing, and inquiries.</p>', unsafe_allow_html=True)

# --- INITIALIZATION (Caching ensures this only runs once) ---
@st.cache_resource
def init_models_and_kb():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=
