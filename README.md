# AI Avatar Tutor
An end-to-end AI tutor pipeline where users speak to the system, which transcribes voice input, retrieves relevant context from uploaded documents via RAG, generates intelligent responses using Groq LLM, and delivers answers as a talking avatar video via HeyGen.
## Architecture
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND (Browser) │ │ ┌──────────┐ ┌──────────────┐
┌──────────┐ ┌───────────┐ │ │ │ Mic Input│ │ Chat History │ │ Avatar │ │
Doc Mgmt │ │ │ │ Recorder │ │ Display │ │ Player │ │ Upload │ │ │ └────┬─────┘
└──────▲───────┘ └────▲─────┘ └─────┬─────┘ │ │ │ │ │ │ │
└───────┼────────────────┼───────────────┼───────────────┼────────┘
│ │ │ │ ▼ │ │ ▼
┌─────────────────────────────────────────────────────────────────┐
│ FASTAPI BACKEND │ │ │ │ ┌─────────────────── PIPELINE
ORCHESTRATOR ─────────────────┐ │ │ │ │ │ │ │ ┌─────────┐
┌─────────┐ ┌─────────┐ ┌─────────┐ │ │ │ │ │ STT │──[Start]│ RAG
│──[Start]│ LLM │──[Start]│ Avatar │ │ │ │ │ │Eleven │ │Retriever│ │ Groq │ │
HeyGen │ │ │ │ │ │Labs │ │ │ │ │ │ │ │ │ │ │ └─────────┘ └────┬────┘
└─────────┘ └─────────┘ │ │ │ │ │ │ │ │
└──────────────────────┼─────────────────────────────────────┘
│ │ │ │ │
┌──────────────────────▼─────────────────────────────────────┐
│ │ │ RAG SUBSYSTEM │ │ │ │ ┌──────────┐ ┌──────────┐
┌────────────┐ │ │ │ │ │Doc Loader│ │ Embedder │ │Vector Store│ │ │ │ │
│PDF/TXT/MD│ │MiniLM-L6 │ │ FAISS │ │ │ │ │ └──────────┘
└──────────┘ └────────────┘ │ │ │
└────────────────────────────────────────────────────────────┘
│ │ │
└──────────────────────────────────────────────────────────────────┘
## Prerequisites
- Python 3.10 or higher
- pip package manager
- A modern web browser with microphone access
- API keys for Groq, ElevenLabs, and HeyGen
## System Requirements
- RAM: Minimum 4GB (8GB recommended for embedding model)
- Storage: 2GB free space for models and vector store
- OS: Linux, macOS, or Windows
## Installation
### 1. Clone the repository
git clone https://github.com/yourusername/ai-avatar-tutor.git
cd ai-avatar-tutor
2. Create a virtual environment
bash
python -m venv venv
source venv/bin/activate # Linux/macOS
### or
venv\Scripts\activate # Windows
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables
bash
cp .env.example .env
### Edit .env with your API keys
5. Get API Keys
Groq API Key
1. Go to https://console.groq.com (https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy it to in
            GROQ_API_KEY .env
ElevenLabs API Key
1. Go to https://elevenlabs.io (https://elevenlabs.io/)
2. Sign up or log in
3. Go to Profile Settings API Keys
4. Copy your API key to in
                     ELEVENLABS_API_KEY .env
5. Choose a voice and copy its ID to
                               ELEVENLABS_VOICE_ID
HeyGen API Key
1. Go to https://app.heygen.com (https://app.heygen.com/)
2. Sign up or log in
3. Go to Settings API
4. Generate and copy your API key to
                                 HEYGEN_API_KEY
5. Choose an avatar and copy its ID to
                                 HEYGEN_AVATAR_ID
6. Add documents to RAG
Place your PDF, TXT, or MD files in the docs/ folder. They will be automatically indexed
on first run.
cp your_document.pdf docs/
7. Run the application
python main.py
The application will be available at http://localhost:8000
Docker Setup
docker-compose up --build
Running Tests
pytest tests/ -v
How to Use
1. Open the web interface at
                         http://localhost:8000
2. Upload study documents using the document management panel
3. Press and hold the microphone button to ask a question
4. Wait for the AI tutor to respond with text and avatar video
5. Continue the conversation with follow-up questions
Troubleshooting
1. "ModuleNotFoundError: No module named
'sentence_transformers'"
pip install sentence-transformers --no-cache-dir
2. "FAISS index not found" error
This happens on first run. The vector store will be created automatically when you upload
documents or if files exist in the docs/ folder.
3. "Microphone access denied"
Ensure your browser has permission to access the microphone. Check browser settings
and use HTTPS or localhost.
4. "Groq API rate limit exceeded"
The free tier has rate limits. Wait 60 seconds and try again, or upgrade your Groq
plan.
5. "HeyGen video generation timeout"
HeyGen video generation can take 30-60 seconds. The system will fall back to
text-only response if it exceeds the timeout. Check your HeyGen API quota.
Architecture Details
Component Technology Purpose
STT ElevenLabs Voice to text transcription
RAG FAISS + MiniLM Document retrieval
LLM Groq (LLaMA 370B) Response generation
Avatar HeyGen Talking avatar video
Backend FastAPI API server
Frontend Vanilla JS User interface
License
MIT License
---