import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain and API Imports
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration Variables
video_id = "uOcKF-aLHyw" 
db_path = "faiss_index"

# Initialize docs container outside scope
docs = []

# ==========================================
# STEP 1: RETRIEVAL & VECTORSTORE SETTINGS
# ==========================================
try:
    # 1. Fetch YouTube Transcript using Instance Object Methods
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    english_transcript = transcript_list.find_transcript(["en"])
    transcript_data = english_transcript.fetch()
    
    # Extract text content carefully from transcript dictionaries
    transcript_chunks = []
    for chunk in transcript_data:
        if isinstance(chunk, dict) and 'text' in chunk:
            text_content = chunk['text']
        else:
            text_content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        transcript_chunks.append(text_content)
        
    transcript = ' '.join(transcript_chunks)
    print("--- Transcript Fetched Successfully ---")

    # 2. Split the text into clean text fragments
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.create_documents([transcript])
    print(f"Split transcript into {len(docs)} document chunks.")

    # 3. Check for API Credentials
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from your environment configuration / .env file.")

except TranscriptsDisabled:
    print("No captions available for this video.")
except Exception as e:
    print(f"An error occurred during extraction: {e}")


# ==========================================
# STEP 2: BUILD PIPELINE IF TRANSCRIPT EXISTS
# ==========================================
if docs:
    print("\nGenerating embeddings and creating FAISS vector database...")
    
    # 1. Generate Embeddings and initialize Database
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save the database locally
    vector_store.save_local(db_path)
    print("--- FAISS Vector Database Saved Successfully Locally ---")
    
    # 2. Build the Retriever Element
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # ==========================================
    # STEP 3: THE AUGMENTATION & LLM CHAIN
    # ==========================================
    # System Prompts specifying deterministic document retrieval constraints
    template = """
    You are an intelligent chatbot assisting a user with questions about a YouTube video transcript. 
    Answer the user's question accurately using ONLY the provided video transcript context below. 
    If you do not know the answer based on the context, state that you cannot find the answer in the video.

    Context from video:
    {context}

    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the LLM (GPT-4o-mini is optimized for fast performance and low cost)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Context aggregation mapping helper
    def format_docs(retrieved_documents):
        return "\n\n".join(doc.page_content for doc in retrieved_documents)
        
    # Constructing the Complete RAG Chain via LCEL Pipes (|)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("--- RAG Processing Chain Built Seamlessly ---")
    
    # ==========================================
    # STEP 4: INTERACTIVE CHAT LOOP
    # ==========================================
    print("\n================== CHATBOT ACTIVE ==================")
    print("Type your questions below regarding the video. Type 'exit' to quit.")
    print("====================================================\n")
    
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Shutting down conversational engine. Goodbye!")
            break
            
        if not user_query:
            continue
            
        print("Bot is thinking...")
        # Stream or invoke directly down the augmentation chain pipes
        response = rag_chain.invoke(user_query)
        print(f"\nAI: {response}\n")
        print("-" * 50)
else:
    print("\nInitialization halted: Could not parse database artifacts since document chunks are completely empty.")
