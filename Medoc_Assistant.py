import chainlit as cl
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import sqlite3
from datetime import datetime
import speech_recognition as sr
import pyttsx3

# Initialize Database for Search History
conn = sqlite3.connect("search_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    query TEXT, 
                    timestamp TEXT)''')
conn.commit()

def save_query(user_query):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO history (query, timestamp) VALUES (?, ?)", (user_query, timestamp))
    conn.commit()

def get_search_history():
    cursor.execute("SELECT query FROM history ORDER BY id DESC LIMIT 10")
    return [row[0] for row in cursor.fetchall()]

def clear_search_history():
    cursor.execute("DELETE FROM history")
    conn.commit()

# Load Medical Knowledge Base
def load_medical_knowledge():
    try:
        loader = TextLoader("d:/langchain-gemma-ollama-chainlit-main/medical_data.txt", encoding="utf-8")
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        print(f"Error loading FAISS: {e}")
        return None

retriever = load_medical_knowledge()

# Text-to-Speech Function (Medoc Speaks)
engine = pyttsx3.init()
def speak_response(response_text):
    engine.say(response_text)
    engine.runAndWait()

# Voice-to-Text Function (User Speaks)
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"🗣️ Recognized: {text}")
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "Speech recognition service is unavailable."

@cl.on_chat_start
async def on_chat_start():
    elements = [cl.Image(name="image1", display="inline", path="medoc_avatar.png")]
    await cl.Message(content="👋 Hello! I am **Medoc**, your AI healthcare assistant. How can I assist you today?", elements=elements).send()
    
    model = Ollama(model="monotykamary/medichat-llama3:latest")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI healthcare assistant named Medoc. You provide reliable medical information but do not diagnose illnesses. Your responses should be concise, clear, and based on verified healthcare data."),
        ("human", "{question}"),
    ])
    
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content.strip()
    save_query(user_query)
    
    if user_query.lower() == "show history":
        history = get_search_history()
        history_text = "\n".join([f"🔹 {query}" for query in history]) if history else "No search history yet."
        await cl.Message(content=history_text).send()
        return
    elif user_query.lower() == "clear history":
        clear_search_history()
        await cl.Message(content="🗑️ Search history cleared!").send()
        return
    elif user_query.lower() == "voice input":
        user_query = recognize_speech()
        await cl.Message(content=f"🗣️ You said: {user_query}").send()

    runnable = cl.user_session.get("runnable")  
    msg = cl.Message(content="⏳ Medoc is thinking...")
    async for chunk in runnable.astream(
        {"question": user_query},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    
    response_text = msg.content  # Get Medoc's response
    speak_response(response_text)  # Medoc speaks
    await msg.send()

# Run Chainlit App
if __name__ == "__main__":
    cl.run("Medoc - AI Healthcare Assistant")