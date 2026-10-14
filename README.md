# 🤖 YouTube Video RAG Chatbot

An intelligent, context-aware chatbot capable of fetching a YouTube video's transcript, processing its textual data, and using a **Retrieval-Augmented Generation (RAG)** pipeline to answer user questions about the video in real-time.

Built using **LangChain Expression Language (LCEL)**, **OpenAI Embeddings**, and **FAISS (Facebook AI Similarity Search)**.

---

## 🚀 How It Works (System Architecture)

This application follows a complete 4-stage RAG data pipeline:

1. **Extraction (Data Ingestion):** The system uses `youtube-transcript-api` to pull the raw subtitles/captions of a specified YouTube video ID.
2. **Chunking & Embedding:** The raw transcript is processed by `RecursiveCharacterTextSplitter` into manageable chunks (1,000 characters) to fit LLM context limits. These chunks are converted into vector representations using OpenAI's `text-embedding-3-small` model.
3. **Vector Storage:** The generated vectors are saved into a local **FAISS** vector database instance, allowing rapid semantic similarity lookups.
4. **Augmented Generation:** When a user poses a question, the FAISS database extracts the most relevant transcript snippets. LangChain constructs a custom prompt injecting these snippets as factual context, passing the augmented query to `gpt-4o-mini` for a precise, deterministic answer.

---

## 🛠️ Tech Stack & Dependencies

- **Orchestration:** LangChain Core & LCEL (LangChain Expression Language)
- **Vector Database:** FAISS (CPU)
- **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- **API Scraping:** Youtube Transcript API

---

## 📦 Getting Started & Installation

### 1. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
touch .env
```
Open the `.env` file and insert your private OpenAI API Key:
```text
OPENAI_API_KEY=sk-proj-yourOpenAiApiKeyHere...
```
*(Note: The `.env` file is protected locally and will be automatically blocked from Git commits via the `.gitignore` rule).*

---

## 🎮 Usage

1. Open `youtubechatbotapp.py` and modify the target `video_id` variable with your preferred YouTube video code:
   ```python
   video_id = "uOcKF-aLHyw" 
   ```
2. Launch the application in your terminal:
   ```bash
   python3 youtubechatbotapp.py
   ```
3. Once the database is successfully built and saved locally, an interactive chat window will open directly inside your terminal. You can query the video until you enter `exit`.

---

## 📂 Project Structure

```text
youtube-chatbot/
│
├── .gitignore          # Prevents pushing .env, cache, or raw indexes
├── .env.example        # Environment variable schema template
├── requirements.txt    # Application package dependencies
├── youtubechatbotapp.py # Main orchestration and interactive script
└── README.md           # Project documentation and architectural breakdown
```
