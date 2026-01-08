# 🤖 FinRobot: Integrated Financial Analyst

**FinRobot** is a production-grade, agentic AI system that combines **advanced reasoning** with **autonomous financial data retrieval**. 

It integrates two powerful systems:
1. **Agentic Reasoning Core**: A structured cognitive pipeline (Planner → Thinker → Verifier) ensuring explainable and hallucinations-free answers.
2. **Autonomous RAG Engine**: A control plane that automatically fetches, indexes, and maintains fresh financial data (SEC filings, Stock Prices, News) from the web.

---

## 🚀 Key Features

- **Autonomous Data Acquisition**: Automatically detects tikers (e.g., "$AAPL") and fetches missing data (10-K, Prices) before answering.
- **Verifiable Reasoning**: Every answer goes through a self-verification loop to check for factual accuracy and compliance.
- **Long-Term Memory**: Remembers user preferences, risk tolerance, and past interactions.
- **Hybrid Retrieval**: Combines structured financial data with unstructured semantic search over filings.

---

## 🗂️ Project Structure

```text
FinRobot_Integrated/
│
├── chatbot_ui.py              # Main Entry Point (Streamlit UI)
├── requirements.txt           # Integrated Dependencies
├── .env                       # API Keys Configuration
│
├── agent/                     # Agentic Cognitive Layer
│   ├── meta_agent.py          # Orchestrator
│   ├── planner.py             # Task Decomposition
│   ├── thinker.py             # Reasoning & Synthesis
│   └── verifier.py            # Safety & Fact Checking
│
├── rag_engine/                # Data & Retrieval Engine
│   ├── src/
│   │   ├── control_plane/     # Data Lifecycle (Fetch/Index)
│   │   ├── inference_plane/   # Read-Only Retrieval
│   │   └── orchestrate.py     # RAG Entry Point
│
├── retrieval/                 # Integration Layer
│   └── pinecone_client.py     # Adapts RAG Engine for the Agent
│
├── memory/                    # User Context & Long-term Memory
├── prompts/                   # System Prompts
└── config/                    # Settings & Rules
```

---

## 🛠️ Installation

1. **Clone & Enter Directory**:
   ```bash
   cd FinRobot_Integrated
   ```

2. **Create Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   Create a `.env` file with the following keys:
   ```env
   # LLM Providers
   OPENAI_API_KEY=sk-...  (or GEMINI_API_KEY)

   # Vector Database
   PINECONE_API_KEY=...
   PINECONE_INDEX_NAME=financial-analysis  (Ensure this index exists)
   PINECONE_ENV=us-east-1

   # RAG Data Sources
   HUGGING_FACE_API_KEY=hf_...  (For embeddings)
   NEWSAPI_KEY=...              (Optional: For news)
   ```

---

## ▶️ Usage

Start the application:
```bash
streamlit run chatbot_ui.py
```

### Example Workflow
1. **Login**: Enter a User ID (e.g., "analyst_01").
2. **Profile**: Set your risk tolerance and explanation depth.
3. **Ask**: *"Analyze the risk factors for Apple based on their latest 10-K."*
4. **Agent Action**:
   - Identifies "AAPL".
   - **RAG Engine**: Checks if 2025 10-K is indexed. If not, downloads from SEC EDGAR and segments it.
   - **Retrieval**: Finds relevant risk sections.
   - **Thinker**: Synthesizes an answer citing specific sections.
   - **Verifier**: Double-checks the claims against the retrieved text.
   - **Response**: Delivers the final answer to you.

---

## 🧠 Architecture Details

### The Agent (Reasoning)
The agent avoids "black box" generation by splitting the process:
- **Planner**: "I need to find Apple's 10-K and look for 'Risk Factors'."
- **Thinker**: Executes the retrieval and drafts a response.
- **Verifier**: "The draft claims revenue grew 5%, but the text says 3%. Correction needed."

### The RAG Engine (Data)
A "Two-Plane" architecture ensures data integrity:
- **Control Plane**: Write-Only. Ensures data is fresh. Mirrors local disk to Pinecone.
- **Inference Plane**: Read-Only. Fast semantic search for the Agent.

---

## 📄 License
MIT License
