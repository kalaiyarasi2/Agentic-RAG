import os
import hashlib
import json
import pickle
from datetime import datetime
from typing import Dict, List, TypedDict, Any
from dotenv import load_dotenv

# --- Core LangChain/LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load Environment Variables ---
load_dotenv()

# =============================================================================
# SECTION 1: AGENTIC INTELLIGENCE COMPONENTS
# =============================================================================

class AgentMemory:
    """Memory system for learning from past interactions"""
    
    def __init__(self, memory_file="agent_memory.pkl"):
        self.memory_file = memory_file
        self.successful_queries = {}
        self.failed_queries = {}
        self.successful_rewrites = {}
        self.grading_patterns = {}
        self.load_memory()
    
    def remember_success(self, original_query: str, final_query: str, strategy: str):
        """Learn from successful query resolutions"""
        self.successful_queries[original_query] = {
            'final_query': final_query,
            'strategy': strategy,
            'timestamp': datetime.now().isoformat(),
            'success_count': self.successful_queries.get(original_query, {}).get('success_count', 0) + 1
        }
        self.save_memory()
    
    def remember_failure(self, query: str, reason: str):
        """Learn from failed attempts"""
        if query not in self.failed_queries:
            self.failed_queries[query] = []
        self.failed_queries[query].append({
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        self.save_memory()
    
    def remember_rewrite_success(self, original: str, rewritten: str):
        """Learn successful rewrite patterns"""
        self.successful_rewrites[original] = rewritten
        self.save_memory()
    
    def get_similar_successful_query(self, query: str) -> str:
        """Find similar previously successful queries"""
        query_lower = query.lower()
        for successful_query, data in self.successful_queries.items():
            if any(word in successful_query.lower() for word in query_lower.split()):
                return data['final_query']
        return None
    
    def save_memory(self):
        """Persist memory to disk"""
        memory_data = {
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'successful_rewrites': self.successful_rewrites,
            'grading_patterns': self.grading_patterns
        }
        with open(self.memory_file, 'wb') as f:
            pickle.dump(memory_data, f)
    
    def load_memory(self):
        """Load memory from disk"""
        try:
            with open(self.memory_file, 'rb') as f:
                memory_data = pickle.load(f)
                self.successful_queries = memory_data.get('successful_queries', {})
                self.failed_queries = memory_data.get('failed_queries', {})
                self.successful_rewrites = memory_data.get('successful_rewrites', {})
                self.grading_patterns = memory_data.get('grading_patterns', {})
        except FileNotFoundError:
            pass

class GoalPlanner:
    """Intelligent goal decomposition and planning"""
    
    def __init__(self, llm):
        self.llm = llm
        self.learned_decompositions = {}
    
    def decompose_complex_query(self, query: str) -> List[str]:
        """Break complex queries into manageable sub-goals"""
        if query in self.learned_decompositions:
            return self.learned_decompositions[query]
        
        prompt = ChatPromptTemplate.from_template(
            "You are a query planning expert. Break down this complex question into 2-3 simpler, focused sub-questions "
            "that can be answered independently and then combined for a complete response.\n\n"
            "Complex Question: {query}\n\n"
            "Sub-questions (one per line):"
        )
        
        decomposer_chain = prompt | self.llm | StrOutputParser()
        result = decomposer_chain.invoke({"query": query})
        
        sub_goals = [line.strip() for line in result.split('\n') if line.strip()]
        self.learned_decompositions[query] = sub_goals
        
        return sub_goals
    
    def should_decompose(self, query: str) -> bool:
        """Decide if query is complex enough to warrant decomposition"""
        complexity_indicators = ['and', 'also', 'moreover', 'furthermore', 'explain', 'compare', 'analyze']
        return any(indicator in query.lower() for indicator in complexity_indicators) and len(query.split()) > 8

class ReinforcementLearner:
    """Learning system for optimizing agent strategies"""
    
    def __init__(self):
        self.strategy_performance = {
            'direct_retrieval': {'successes': 0, 'attempts': 0},
            'rewrite_once': {'successes': 0, 'attempts': 0},
            'rewrite_twice': {'successes': 0, 'attempts': 0},
            'decompose_query': {'successes': 0, 'attempts': 0}
        }
        self.grading_thresholds = {
            'strict': 0.9,
            'moderate': 0.6,
            'lenient': 0.3
        }
        self.current_threshold = 'moderate'
    
    def update_strategy_performance(self, strategy: str, success: bool):
        """Learn from strategy outcomes"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {'successes': 0, 'attempts': 0}
        
        self.strategy_performance[strategy]['attempts'] += 1
        if success:
            self.strategy_performance[strategy]['successes'] += 1
    
    def get_best_strategy(self) -> str:
        """Dynamically choose best performing strategy"""
        best_strategy = 'direct_retrieval'
        best_rate = 0
        
        for strategy, perf in self.strategy_performance.items():
            if perf['attempts'] > 0:
                rate = perf['successes'] / perf['attempts']
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy
        
        return best_strategy
    
    def adapt_grading_threshold(self, recent_failures: int):
        """Dynamically adjust grading strictness based on performance"""
        if recent_failures > 3:
            self.current_threshold = 'lenient'
        elif recent_failures > 1:
            self.current_threshold = 'moderate'
        else:
            self.current_threshold = 'strict'
        
        return self.grading_thresholds[self.current_threshold]

# =============================================================================
# SECTION 2: ENHANCED AGENTIC STATE
# =============================================================================

class AgenticState(TypedDict):
    question: str
    original_question: str
    documents: List[str]
    generation: str
    rewrite_count: int
    strategy_used: str
    sub_goals: List[str]
    current_sub_goal: int
    confidence_score: float
    learning_feedback: Dict[str, Any]

# =============================================================================
# SECTION 3: FULLY AGENTIC RAG IMPLEMENTATION
# =============================================================================

def create_fully_agentic_rag(collection_name: str, db_path: str):
    """Create the fully agentic RAG system with learning and memory"""
    
    # Initialize agentic components
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
    memory = AgentMemory()
    goal_planner = GoalPlanner(llm)
    learner = ReinforcementLearner()
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=CohereEmbeddings(model="embed-english-v3.0"),
        persist_directory=db_path,
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # --- AGENTIC AGENTS WITH LEARNING CAPABILITIES ---
    
    def strategic_planner_agent(state):
        """STRATEGIC PLANNER: Analyzes query and plans approach"""
        print("🧠 STRATEGIC PLANNER AGENT: ANALYZING QUERY & PLANNING APPROACH")
        
        question = state["question"]
        original_question = state.get("original_question", question)
        
        # Check memory for similar successful queries
        similar_query = memory.get_similar_successful_query(question)
        if similar_query:
            print(f"   🎯 MEMORY: Found similar successful query pattern")
            return {
                "question": similar_query,
                "original_question": original_question,
                "strategy_used": "memory_guided",
                "rewrite_count": 0
            }
        
        # Decide if query needs decomposition
        if goal_planner.should_decompose(question):
            print(f"   📋 PLANNING: Complex query detected, decomposing...")
            sub_goals = goal_planner.decompose_complex_query(question)
            return {
                "question": question,
                "original_question": original_question,
                "sub_goals": sub_goals,
                "current_sub_goal": 0,
                "strategy_used": "decompose_query"
            }
        
        # Use best learned strategy
        best_strategy = learner.get_best_strategy()
        print(f"   🎖️ LEARNING: Using best performing strategy: {best_strategy}")
        
        return {
            "question": question,
            "original_question": original_question,
            "strategy_used": best_strategy,
            "rewrite_count": 0
        }
    
    def adaptive_retriever_agent(state):
        """ADAPTIVE RETRIEVER: Learns optimal retrieval patterns"""
        print("🔍 ADAPTIVE RETRIEVER AGENT: EXECUTING LEARNED RETRIEVAL PATTERNS")
        
        question = state["question"]
        strategy = state.get("strategy_used", "direct_retrieval")
        
        # Adapt retrieval based on strategy
        if strategy == "decompose_query" and "sub_goals" in state:
            current_goal_idx = state.get("current_sub_goal", 0)
            if current_goal_idx < len(state["sub_goals"]):
                question = state["sub_goals"][current_goal_idx]
                print(f"   🎯 Processing sub-goal {current_goal_idx + 1}: {question}")
        
        documents = retriever.invoke(question)
        docs_content = [doc.page_content for doc in documents]
        
        print(f"   📄 Retrieved {len(docs_content)} documents using {strategy}")
        
        return {"documents": docs_content, "question": question}
    
    def learning_grader_agent(state):
        """LEARNING GRADER: Adapts grading criteria based on performance"""
        print("⚖️ LEARNING GRADER AGENT: ADAPTIVE QUALITY ASSESSMENT")
        
        question = state["question"]
        documents = state["documents"]
        
        # Get adaptive threshold
        recent_failures = len(memory.failed_queries.get(question, []))
        threshold = learner.adapt_grading_threshold(recent_failures)
        
        print(f"   📊 Using {learner.current_threshold} grading (threshold: {threshold})")
        
        # FIXED: Enhanced grading with proper escaping
        prompt = ChatPromptTemplate.from_template(
            "You are an adaptive document grader. Rate document relevance on a scale of 0.0-1.0 for the given question. "
            f"Consider partial matches and contextual relevance. Current threshold: {threshold}\n\n"
            "QUESTION: {question}\n\nDOCUMENT: {document}\n\nRelevance Score (0.0-1.0):"
        )
        
        grader_chain = prompt | llm | StrOutputParser()
        
        relevant_docs = []
        confidence_scores = []
        
        for i, doc in enumerate(documents):
            try:
                score_str = grader_chain.invoke({
                    "question": question, 
                    "document": doc
                })
                # Extract numeric score from response
                score = 0.0
                for word in score_str.split():
                    try:
                        potential_score = float(word.strip('.,()[]'))
                        if 0.0 <= potential_score <= 1.0:
                            score = potential_score
                            break
                    except ValueError:
                        continue
                
                confidence_scores.append(score)
                
                if score >= threshold:
                    print(f"   ✅ DOCUMENT {i+1}: RELEVANT (Score: {score:.2f})")
                    relevant_docs.append(doc)
                else:
                    print(f"   ❌ DOCUMENT {i+1}: NOT RELEVANT (Score: {score:.2f})")
            except Exception as e:
                # Fallback to simple yes/no grading
                simple_prompt = ChatPromptTemplate.from_template(
                    "Is this document relevant to the question? Answer only 'yes' or 'no'.\n"
                    "QUESTION: {question}\n\nDOCUMENT: {document}\n\nAnswer:"
                )
                simple_chain = simple_prompt | llm | StrOutputParser()
                grade = simple_chain.invoke({"question": question, "document": doc})
                
                if 'yes' in grade.lower():
                    relevant_docs.append(doc)
                    confidence_scores.append(0.6)
                    print(f"   ✅ DOCUMENT {i+1}: RELEVANT (fallback)")
                else:
                    confidence_scores.append(0.2)
                    print(f"   ❌ DOCUMENT {i+1}: NOT RELEVANT (fallback)")
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            "documents": relevant_docs, 
            "confidence_score": avg_confidence
        }
    
    def intelligent_decision_agent(state):
        """INTELLIGENT DECISION: Dynamic routing with learning"""
        print("🤔 INTELLIGENT DECISION AGENT: ADAPTIVE ROUTING")
        
        rewrite_count = state.get("rewrite_count", 0)
        confidence = state.get("confidence_score", 0)
        strategy = state.get("strategy_used", "direct_retrieval")
        
        if not state["documents"]:
            if rewrite_count >= 2:
                print("   🛑 DECISION: Max rewrites reached → FALLBACK")
                learner.update_strategy_performance(strategy, False)
                memory.remember_failure(state["question"], "max_rewrites_reached")
                return "generate_fallback"
            else:
                print(f"   🔄 DECISION: No relevant docs (confidence: {confidence:.2f}) → ADAPTIVE REWRITE")
                return "adaptive_rewrite"
        else:
            print(f"   ✅ DECISION: Found {len(state['documents'])} relevant docs → GENERATE")
            learner.update_strategy_performance(strategy, True)
            return "generate"
    
    def adaptive_rewriter_agent(state):
        """ADAPTIVE REWRITER: Learns successful rewrite patterns"""
        print("✏️ ADAPTIVE REWRITER AGENT: LEARNING-BASED QUERY OPTIMIZATION")
        
        question = state["question"]
        rewrite_count = state.get("rewrite_count", 0) + 1
        original_question = state.get("original_question", question)
        
        # Check for learned rewrite patterns
        if question in memory.successful_rewrites:
            learned_rewrite = memory.successful_rewrites[question]
            print(f"   🎓 LEARNING: Using previously successful rewrite pattern")
            return {
                "question": learned_rewrite,
                "rewrite_count": rewrite_count,
                "original_question": original_question
            }
        
        # Adaptive rewrite strategies
        strategies = [
            "Make the query more specific with technical keywords",
            "Simplify the query to basic concepts", 
            "Rephrase focusing on key entities and relationships"
        ]
        
        strategy_desc = strategies[min(rewrite_count - 1, len(strategies) - 1)]
        
        prompt = ChatPromptTemplate.from_template(
            f"You are an adaptive query rewriter. {strategy_desc} for this question: {{question}}\n"
            "Rewritten Question:"
        )
        
        rewriter_chain = prompt | llm | StrOutputParser()
        better_question = rewriter_chain.invoke({"question": question})
        
        print(f"   🔄 REWRITE #{rewrite_count}: '{question}' → '{better_question}'")
        
        return {
            "question": better_question,
            "rewrite_count": rewrite_count,
            "original_question": original_question
        }
    
    def learning_generator_agent(state):
        """LEARNING GENERATOR: Improves generation based on feedback"""
        print("📝 LEARNING GENERATOR AGENT: ADAPTIVE ANSWER SYNTHESIS")
        
        question = state["question"]
        original_question = state.get("original_question", question)
        documents = state["documents"]
        confidence = state.get("confidence_score", 1.0)
        strategy = state.get("strategy_used", "direct_retrieval")
        
        # Adaptive generation based on confidence
        if confidence > 0.8:
            generation_style = "detailed and comprehensive"
        elif confidence > 0.5:
            generation_style = "focused on available information"
        else:
            generation_style = "cautious with uncertainty indicators"
        
        prompt = ChatPromptTemplate.from_template(
            f"Generate a {generation_style} answer based on the provided context. "
            f"Confidence level: {confidence:.2f}\n\n"
            "Context: {context}\nQuestion: {question}\nAnswer:"
        )
        
        generator_chain = prompt | llm | StrOutputParser()
        generation = generator_chain.invoke({
            "context": "\n\n".join(documents), 
            "question": original_question
        })
        
        # Learn from successful generation
        memory.remember_success(original_question, question, strategy)
        learner.update_strategy_performance(strategy, True)
        
        print(f"   ✅ Generated answer with {confidence:.2f} confidence using {strategy}")
        
        return {"generation": generation}
    
    def adaptive_fallback_agent(state):
        """ADAPTIVE FALLBACK: Learning from failures"""
        print("🔄 ADAPTIVE FALLBACK AGENT: LEARNING FROM FAILURE")
        
        question = state.get("original_question", state["question"])
        strategy = state.get("strategy_used", "unknown")
        
        # Learn from failure
        memory.remember_failure(question, f"strategy_{strategy}_failed")
        learner.update_strategy_performance(strategy, False)
        
        prompt = ChatPromptTemplate.from_template(
            "The agentic learning system attempted multiple strategies but couldn't find relevant information "
            "in the knowledge base for: '{question}'. This failure has been recorded for future learning. "
            "Provide a helpful response explaining the limitation.\n"
            "Question: {question}\nResponse:"
        )
        
        fallback_chain = prompt | llm | StrOutputParser()
        generation = fallback_chain.invoke({"question": question})
        
        return {"generation": generation}
    
    # --- BUILD FULLY AGENTIC WORKFLOW ---
    workflow = StateGraph(AgenticState)
    
    workflow.add_node("strategic_planner", strategic_planner_agent)
    workflow.add_node("adaptive_retriever", adaptive_retriever_agent)
    workflow.add_node("learning_grader", learning_grader_agent)
    workflow.add_node("adaptive_rewrite", adaptive_rewriter_agent)
    workflow.add_node("learning_generator", learning_generator_agent)
    workflow.add_node("adaptive_fallback", adaptive_fallback_agent)
    
    workflow.set_entry_point("strategic_planner")
    workflow.add_edge("strategic_planner", "adaptive_retriever")
    workflow.add_edge("adaptive_retriever", "learning_grader")
    workflow.add_conditional_edges(
        "learning_grader",
        intelligent_decision_agent,
        {
            "adaptive_rewrite": "adaptive_rewrite",
            "generate": "learning_generator", 
            "generate_fallback": "adaptive_fallback"
        },
    )
    workflow.add_edge("adaptive_rewrite", "adaptive_retriever")
    workflow.add_edge("learning_generator", END)
    workflow.add_edge("adaptive_fallback", END)
    
    return workflow.compile()

# =============================================================================
# SECTION 4: DATA INGESTION & MAIN EXECUTION
# =============================================================================

def ingest_data(file_path: str, collection_name: str, db_path: str):
    """Data ingestion function"""
    print(f"\n--- Starting data ingestion for: {file_path} ---")
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    if not splits:
        return False
    
    doc_ids = [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in splits]
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        ids=doc_ids,
        collection_name=collection_name,
        persist_directory=db_path
    )
    
    print(f"--- Data ingestion complete ---")
    return True

def main():
    """Main function with fully agentic capabilities"""
    print("🧠 FULLY AGENTIC RAG SYSTEM WITH LEARNING & MEMORY")
    print("="*70)
    
    # Check API keys
    if any(not os.getenv(var) for var in ["GROQ_API_KEY", "COHERE_API_KEY"]):
        print("❌ Error: Missing API keys")
        return
    
    # Get document
    file_path = input("Enter document path: ").strip()
    if not os.path.exists(file_path):
        print("❌ File not found")
        return
    
    # Setup
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    collection_name = f"agentic_rag_{file_hash}"
    db_path = f"./chroma_db_{file_hash}"
    
    # Process document
    if not os.path.exists(db_path):
        if not ingest_data(file_path, collection_name, db_path):
            return
    
    # Create fully agentic system
    app = create_fully_agentic_rag(collection_name, db_path)
    
    # Interactive loop with learning
    while True:
        try:
            question = input("\n🎯 Enter your question (or 'exit'): ").strip()
            if question.lower() in ['exit', 'quit']:
                break
            
            print("\n" + "="*70)
            print("🧠 AGENTIC LEARNING SYSTEM PROCESSING")
            print("="*70)
            
            inputs = {"question": question}
            
            for output in app.stream(inputs, {"recursion_limit": 20}):
                for key, value in output.items():
                    print(f"--- Completed: {key.upper()} ---")
            
            final_answer = value.get("generation", "No answer generated.")
            
            print("\n" + "="*70)
            print("🎉 AGENTIC RESULT WITH LEARNING")
            print("="*70)
            print(final_answer)
            print("="*70)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
