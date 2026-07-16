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

# --- CUSTOM CSS FOR BRANDING & TYPOGRAPHY ---
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
    </style>
""", unsafe_allow_html=True)

# --- BRANDED TITLE & SUBTEXT ---
st.markdown('<p class="therapy-title">The Sunday Therapist Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="therapy-caption">Hello and welcome to the automated assistance chatbot for treatments, pricing, and inquiries.</p>', unsafe_allow_html=True)

# --- DEFINE CUSTOM AVATARS (With Fallbacks) ---
# If your images are uploaded to GitHub, Streamlit will load them. Otherwise, it defaults to friendly emojis.
ASSISTANT_AVATAR = "logo.png" if os.path.exists("logo.png") else "🌿"
USER_AVATAR = "customer_avatar.png" if os.path.exists("user_avatar.png") else "👤"

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

def find_csv_match(user_prompt: str, threshold=0.75):
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
    # If there is no chat history yet, provide the updated streamlined greeting
    if not st.session_state.chat_session_history:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi. How can I help  \n\n"
                "**Top 5 questions**"
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
