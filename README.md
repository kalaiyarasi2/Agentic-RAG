# Agentic-RAG
Retrieval-Augmented Generation system built with LangGraph that employs multiple intelligent agents working together to process documents and answer queries. The system learns from past interactions, adapts its strategies, and continuously improves performance through memory and reinforcement learning.
🔍 What Is This?
This project implements an autonomous, self-improving RAG agent that mimics human-like reasoning and learning. Unlike traditional RAG systems, this agent:

🧠 Plans strategically using goal decomposition
📚 Learns from past successes and failures
🔄 Dynamically rewrites queries based on performance
⚖️ Adaptively grades document relevance
🤖 Uses reinforcement learning to optimize its strategies
💾 Persists memory across sessions for continuous improvement
It's ideal for complex, real-world knowledge retrieval where static pipelines fail.

# Features
Feature    -     Description
✅ Strategic Goal Planning   -   Breaks complex questions into sub-goals for better reasoning
✅ Adaptive Query Rewriting  -   Learns how to rephrase queries for optimal retrieval
✅ Memory-Powered Learning   -   Remembers successful patterns and avoids past mistakes
✅ Dynamic Strategy Selection - Chooses best retrieval/generation approach using RL
✅ Confidence-Aware Generation - Adjusts output style based on retrieval confidence
✅ Fallback with Explanation - Gracefully handles unknown queries and learns from failure
✅ Persistent State          -    Saves agent memory between runs using pickle
✅ Modular LangGraph Workflow  -  Built with langgraph" for observable, controllable execution"

# Technologies Used
LangChain + LangGraph – For building agentic workflows
ChatGroq (Llama 3.1) – Fast, powerful LLM backend
Cohere Embeddings – High-quality semantic search
ChromaDB – Lightweight vector database
Unstructured.io – Document parsing (PDF, DOCX, TXT, etc.)
Reinforcement Learning – Strategy optimization via success tracking
Pickle + JSON – Persistent agent memory

# Quick Start
1. Clone the Repository
  git clone https://github.com/yourusername/fully-agentic-rag.git
  cd fully-agentic-rag
2. Install Dependencies
   pip install -r requirements.txt
3. Set Up API Keys
Create a .env file in the root directory:
  GROQ_API_KEY=your_groq_api_key_here
  COHERE_API_KEY=your_cohere_api_key_here
4. Run the System
   python main.py

  You’ll be prompted to enter a document path (PDF, TXT, DOCX supported via unstructured):
  Enter document path: ./sample_document.pdf
Then ask questions:

  🎯 Enter your question (or 'exit'): Explain the key findings in this paper.

The agent will process your query with full visibility into its internal reasoning steps.

🗂️ Project Structure

fully-agentic-rag/
│
├── main.py                 # Main executable with interactive loop
├── .env                    # Environment variables (ignored in git)
├── requirements.txt        # Python dependencies
├── agent_memory.pkl        # Persistent agent memory (auto-generated)
├── chroma_db_<hash>/       # Vector database per document (auto-generated)
│
└── README.md

🧪 Example Output

🧠 AGENTIC LEARNING SYSTEM PROCESSING
============================================================
🧠 STRATEGIC PLANNER AGENT: ANALYZING QUERY & PLANNING APPROACH
   🎖️ LEARNING: Using best performing strategy: direct_retrieval
🔍 ADAPTIVE RETRIEVER AGENT: EXECUTING LEARNED RETRIEVAL PATTERNS
   📄 Retrieved 4 documents using direct_retrieval
⚖️ LEARNING GRADER AGENT: ADAPTIVE QUALITY ASSESSMENT
   📊 Using moderate grading (threshold: 0.6)
   ✅ DOCUMENT 1: RELEVANT (Score: 0.87)
   ❌ DOCUMENT 2: NOT RELEVANT (Score: 0.21)
🤔 INTELLIGENT DECISION AGENT: ADAPTIVE ROUTING
   ✅ DECISION: Found 1 relevant docs → GENERATE
📝 LEARNING GENERATOR AGENT: ADAPTIVE ANSWER SYNTHESIS
   ✅ Generated answer with 0.87 confidence using direct_retrieval
============================================================
🎉 AGENTIC RESULT WITH LEARNING
============================================================
The key findings indicate that...


🌱 Future Enhancements
Add support for multi-modal documents
Integrate feedback loops from users
Enable cross-document reasoning
Implement neural memory (vectorized memory recall)
Add web search fallback for open-domain queries
Visualize agent decision trees
📄 License
MIT License. See LICENSE for details.

🙌 Acknowledgments
LangChain & LangGraph – For enabling modular, agentic AI workflows
Groq & Cohere – For high-performance LLM and embedding APIs
ChromaDB – For lightweight, fast vector storage
Unstructured.io – For seamless document ingestion
💬 Feedback & Contributions
Contributions are welcome! Open an issue or PR for:

Bug fixes
New agent strategies
Improved grading logic
UI/UX enhancements (e.g., Streamlit frontend)
Let’s build smarter, self-learning agents together! 🚀
