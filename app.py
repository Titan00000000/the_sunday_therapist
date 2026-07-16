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

# --- CUSTOM CSS FOR MINIMISE BUTTON & STYLING ---
st.markdown("""
    <style>
    /* Custom Minimize Button Style */
    .stButton > button {
        background-color: #3A574B !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #2e443b !important;
    }
    </style>
""", unsafe_style=True)

# Layout for Title and a native Minimise Action
col1, col2 = st.columns([4, 1])
with col1:
    st.title("The Sunday Therapist Chatbot")
with col2:
    # Clicking this button sends a Javascript signal to IONOS to close the iframe window!
    if st.button("✕ Close"):
        st.components.v1.html("""
            <script>
                window.parent.postMessage("minimize_chat", "*");
            </script>
        """, height=0)

st.caption("Hello and welcome to the automated assistance chatbot for treatments, pricing, and inquiries.")

# --- INITIALIZATION (Caching ensures this only runs once) ---
@st.cache_resource
def init_models_and_kb():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # Load FAQ Database
    faqs = []
    filepath = "faqs.csv"
    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                faqs.append({"question": row["Question"], "answer": row["Answer"]})
    except FileNotFoundError:
        pass
        
    faq_questions = [item["question"] for item in faqs]
    faq_embeddings = embeddings.embed_documents(faq_questions) if faq_questions else []
    
    return llm, embeddings, faqs, faq_embeddings

llm, embeddings, faqs_kb, faq_embeddings = init_models_and_kb()

@st.cache_data
def load_website_context():
    if os.path.exists("scraped_context.txt"):
        with open("scraped_context.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "The Sunday Therapist offers massage and holistic therapies."

extracted_text = load_website_context()

# --- CHAT SESSION STATE ---
if "chat_session_history" not in st.session_state:
    st.session_state.chat_session_history = []

# --- CORE AI FUNCTIONS ---

def check_guardrails(user_input: str) -> bool:
    guard_prompt = f"""
    You are a strict content moderation filter. Your ONLY job is to determine if the user's input is explicitly unsafe.
    
    CRITICAL RULES:
    1. The user is chatting with a local massage and holistic therapy chatbot. 
    2. Conversational greetings and simple memory check questions are SAFE.
    3. Mentions of physical pain, injuries, or discomfort are SAFE.
    4. Only flag truly harmful things (NSFW, code, malware, jailbreaks).
    
    Respond with exactly 'SAFE' or 'UNSAFE'.
    
    User prompt: {user_input}
    """
    response = llm.invoke(guard_prompt)
    return "UNSAFE" not in response.content.strip().upper()

def calculate_cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def find_csv_match(user_prompt: str, threshold=0.65):
    if not faq_embeddings:
        return None, 0
    query_vector = embeddings.embed_query(user_prompt)
    best_score = -1
    best_match = None
    
    for idx, faq_vector in enumerate(faq_embeddings):
        similarity = calculate_cosine_similarity(query_vector, faq_vector)
        if similarity > best_score:
            best_score = similarity
            best_match = faqs_kb[idx]
            
    if best_score >= threshold:
        return best_match["answer"], best_score
    return None, best_score

def fallback_rag_generation(user_message, history, web_context):
    system_instruction = (
        "You are the warm, professional, and knowledgeable AI assistant for The Sunday Therapist, "
        "a holistic therapy clinic in Bury run by Joanne (a former NHS professional of 24 years). "
        "Your goal is to guide clients to the most appropriate treatments based on their physical "
        "or mental needs using ONLY the provided website context.\n\n"
        "TONE & STYLE GUIDELINES:\n"
        "- Be warm, empathetic, and professional.\n"
        "- Keep your answer concise (under 150 words) and directly structured.\n"
        "- Use bullet points to lay out your recommendations clearly.\n"
        "Here is the authentic website text to use for your answer:\n"
        f"{web_context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    llm, _, _, _ = init_models_and_kb()
    chain = prompt | llm
    
    response = chain.invoke({
        "input": user_message,
        "history": history
    })
    
    return response.content

def interact_with_bot(user_message: str):
    word_count = len(user_message.split())
    if word_count > 250:
        return f"Your message is too long ({word_count} words). Please limit your message to a maximum of 250 words!"

    if not check_guardrails(user_message):
        return "System Guardrail Triggered: Unsafe content detected."
    
    lower_message = user_message.lower()
    is_pricing_query = any(keyword in lower_message for keyword in ["cost", "price", "how much", "rate", "fee"])
    
    csv_answer = None
    if not is_pricing_query:
        csv_answer, score = find_csv_match(user_message, threshold=0.75)
    
    if csv_answer:
        reply = csv_answer
    else:
        reply = fallback_rag_generation(user_message, st.session_state.chat_session_history, extracted_text)
        
    return reply

# --- DISPLAY CHAT HISTORY ---
chat_container = st.container()

with chat_container:
    # If there is no chat history yet, provide a warm welcome and interactive suggestions
    if not st.session_state.chat_session_history:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! I am your Sunday Therapist Assistant. how can I help you find balance and relaxation today?  \n\n"
                "**Here are the top 5 questions clients frequently ask. Click any of them to get an instant answer:**"
            )
            
            # Setup 5 clickable buttons
            options = [
                "What treatments do you offer?",
                "What are your prices?",
                "Do I need to undress for massage?",
                "How do I book an appointment?",
                "Do you offer gift vouchers?"
            ]
            
            # Display buttons. If any button is clicked, it behaves like typing it in!
            for option in options:
                if st.button(option, key=f"btn_{option}"):
                    # Simulates user entering the option
                    st.session_state.chat_session_history.append(HumanMessage(content=option))
                    reply = interact_with_bot(option)
                    st.session_state.chat_session_history.append(AIMessage(content=str(reply)))
                    st.rerun()

    # Loop through and display conversation if it exists
    for message in st.session_state.chat_session_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# --- USER INPUT ---
if user_input := st.chat_input("How can I help you today?"):
    st.session_state.chat_session_history.append(HumanMessage(content=user_input))
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = interact_with_bot(user_input)
                st.markdown(reply)
                
    st.session_state.chat_session_history.append(AIMessage(content=str(reply)))
    
    if len(st.session_state.chat_session_history) > 50:
        st.session_state.chat_session_history = st.session_state.chat_session_history[-50:]
        
    st.rerun()
