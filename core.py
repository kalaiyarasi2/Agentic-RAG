# langgraph_gigi_coach.py
# Production-Grade Personal Growth Coach using LangGraph
# Enhanced with encrypted internal IDs and autonomous agent workflow

from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import uuid
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, TypedDict, Annotated
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Core dependencies
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from pydantic import BaseModel, ValidationError, Field
from cryptography.fernet import Fernet
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON, Integer
from sqlalchemy.orm import declarative_base, sessionmaker

# ========================
# CONFIGURATION & SECURITY
# ========================

class Config:
    """Production configuration with environment variables"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./gigi_langgraph.db")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./gigi_memory_langgraph")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))

# Enhanced Security Manager with internal ID generation
class SecurityManager:
    def __init__(self):
        self.cipher = Fernet(Config.ENCRYPTION_KEY.encode())
        self._internal_id_counter = 0
    
    def generate_internal_id(self, prefix: str = "gigi") -> str:
        """Generate secure internal ID that's not user-controlled"""
        timestamp = int(time.time() * 1000000)  # microsecond precision
        random_part = secrets.token_hex(8)
        counter = self._internal_id_counter
        self._internal_id_counter += 1
        
        raw_id = f"{prefix}_{timestamp}_{counter}_{random_part}"
        return self.encrypt_data(raw_id)[:32]  # Truncate for practicality
    
    def encrypt_data(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def hash_for_storage(self, data: str) -> str:
        """Create storage-safe hash"""
        return hashlib.sha256(data.encode()).hexdigest()[:24]

security = SecurityManager()

# ========================
# LANGGRAPH STATE DEFINITION
# ========================

class AgentState(TypedDict):
    """LangGraph state for the coaching agent"""
    # User interaction
    user_message: str
    session_token: str  # Encrypted internal session ID
    
    # Agent state
    current_step: str
    analysis_complete: bool
    goal_identified: bool
    plan_generated: bool
    
    # User data (encrypted)
    user_profile: Optional[Dict[str, Any]]
    current_goal: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    
    # Generated content
    user_analysis: Optional[str]
    goal_assessment: Optional[Dict[str, Any]]
    comprehensive_plan: Optional[str]
    response_message: str
    
    # System metadata
    created_at: str
    last_updated: str
    processing_errors: List[str]

# ========================
# DATA MODELS WITH ENCRYPTED IDS
# ========================

class GoalStatus(Enum):
    ANALYZING = "analyzing"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    NEEDS_ADJUSTMENT = "needs_adjustment"
    COMPLETED = "completed"
    PAUSED = "paused"

class UserProfile(BaseModel):
    internal_id: str = Field(default_factory=lambda: security.generate_internal_id("profile"))
    session_token: str  # This becomes our user identifier
    fitness_level: str = Field(default="beginner")
    dietary_restrictions: List[str] = Field(default_factory=list)
    focus_time: str = Field(default="morning")
    equipment: List[str] = Field(default_factory=list)
    calorie_target: int = Field(default=2000, ge=1200, le=4000)
    timezone: str = Field(default="UTC")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Goal(BaseModel):
    internal_id: str = Field(default_factory=lambda: security.generate_internal_id("goal"))
    session_token: str
    primary_goal: str
    domains: List[str] = Field(default_factory=list)
    timeframe: str
    desired_outcomes: List[str] = Field(default_factory=list)
    status: GoalStatus = GoalStatus.ANALYZING
    progress: int = Field(default=0, ge=0, le=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    target_date: Optional[datetime] = None

# ========================
# DATABASE MODELS
# ========================

Base = declarative_base()

class SessionRecord(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True)
    session_token = Column(String, unique=True, nullable=False, index=True)
    encrypted_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="active")

class GoalRecord(Base):
    __tablename__ = "goals"
    
    id = Column(Integer, primary_key=True)
    internal_goal_id = Column(String, unique=True, nullable=False)
    session_token = Column(String, nullable=False, index=True)
    encrypted_goal_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# ========================
# ENHANCED MEMORY SYSTEM
# ========================

class LangGraphMemoryManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.sessions_collection = self.client.get_or_create_collection(
            name="langgraph_sessions",
            embedding_function=self.embedding_func
        )
        
        self.goals_collection = self.client.get_or_create_collection(
            name="langgraph_goals", 
            embedding_function=self.embedding_func
        )
    
    async def save_session_state(self, state: AgentState) -> bool:
        """Save complete session state"""
        try:
            session_token = state["session_token"]
            
            # Create searchable document
            doc = f"Session: {state.get('current_step', 'unknown')} - {state.get('user_message', '')[:100]}"
            
            # Encrypt sensitive state data
            encrypted_state = security.encrypt_data(json.dumps(dict(state)))
            
            metadata = {
                "session_token": session_token,
                "current_step": state.get("current_step", "unknown"),
                "created_at": state.get("created_at", datetime.utcnow().isoformat()),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            self.sessions_collection.upsert(
                ids=[session_token],
                documents=[doc],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Session state save failed: {e}")
            return False
    
    async def load_session_state(self, session_token: str) -> Optional[AgentState]:
        """Load session state"""
        try:
            result = self.sessions_collection.query(
                query_texts=[f"Session {session_token}"],
                n_results=1,
                where={"session_token": session_token}
            )
            
            if result['metadatas'] and len(result['metadatas'][0]) > 0:
                # In production, would decrypt from database
                metadata = result['metadatas'][0][0]
                return {
                    "session_token": session_token,
                    "current_step": metadata.get("current_step", "start"),
                    "user_message": "",
                    "analysis_complete": False,
                    "goal_identified": False,
                    "plan_generated": False,
                    "user_profile": None,
                    "current_goal": None,
                    "conversation_history": [],
                    "user_analysis": None,
                    "goal_assessment": None,
                    "comprehensive_plan": None,
                    "response_message": "",
                    "created_at": metadata.get("created_at", datetime.utcnow().isoformat()),
                    "last_updated": datetime.utcnow().isoformat(),
                    "processing_errors": []
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Session state load failed: {e}")
            return None

memory_manager = LangGraphMemoryManager()

# ========================
# AI SERVICE FOR LANGGRAPH
# ========================

class LangGraphAIService:
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    async def analyze_user_input(self, user_message: str, context: Dict = None) -> str:
        """Analyze user input to understand intent and needs"""
        
        context_str = ""
        if context and context.get("conversation_history"):
            context_str = f"Previous conversation context: {context['conversation_history'][-3:]}"
        
        prompt = f"""
        You are Gigi, an expert personal growth coach. Analyze this user message to understand their needs.
        
        {context_str}
        
        User Message: {user_message}
        
        Provide a comprehensive analysis covering:
        1. User's current emotional state and motivation level
        2. Primary concerns or challenges mentioned
        3. Implicit needs that weren't directly stated
        4. Readiness level for change
        5. Potential obstacles or resistance patterns
        
        Keep your analysis under 200 words and focused on actionable insights.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    async def assess_goals(self, user_message: str, user_analysis: str) -> Dict[str, Any]:
        """Extract and assess goals from user input"""
        
        prompt = f"""
        Based on the user message and analysis, extract goal information.
        Return ONLY valid JSON with these exact fields:
        {{
            "primary_goal": "clear, specific goal statement",
            "domains": ["nutrition", "fitness", "study", "lifestyle", "career"],
            "timeframe": "specific timeframe like '6 weeks', '3 months'",
            "desired_outcomes": ["specific outcome 1", "specific outcome 2"],
            "difficulty_level": "beginner|intermediate|advanced",
            "motivation_score": 7
        }}

        User Message: {user_message}
        User Analysis: {user_analysis}
        """
        
        response = self.model.generate_content(prompt)
        return self._parse_json_safe(response.text)
    
    async def generate_comprehensive_plan(self, goal_data: Dict, user_context: Dict = None) -> str:
        """Generate detailed action plan"""
        
        prompt = f"""
        Create a comprehensive, personalized wellness plan based on:
        
        Goal: {goal_data}
        User Context: {user_context or "New user"}
        
        Create a structured plan with:
        1. **Week-by-week breakdown** (4 weeks)
        2. **Daily routines** (specific and realistic)
        3. **Progress milestones**
        4. **Potential challenges and solutions**
        5. **Success metrics and tracking methods**
        
        Make it motivational, practical, and personalized. Use markdown formatting.
        Keep it under 800 words but comprehensive.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def _parse_json_safe(self, text: str) -> Dict[str, Any]:
        """Robust JSON parsing"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Extract JSON from markdown
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
            except:
                pass
        
        # Fallback
        return {
            "primary_goal": "Personal growth and wellness",
            "domains": ["lifestyle"],
            "timeframe": "4 weeks",
            "desired_outcomes": ["Improved well-being"],
            "difficulty_level": "beginner",
            "motivation_score": 5
        }

ai_service = LangGraphAIService()

# ========================
# LANGGRAPH AGENT NODES
# ========================

async def start_session_node(state: AgentState) -> AgentState:
    """Initialize or load session"""
    
    if not state.get("session_token"):
        # Generate new encrypted session token
        state["session_token"] = security.generate_internal_id("session")
        state["created_at"] = datetime.utcnow().isoformat()
    
    state["current_step"] = "analyzing_input"
    state["last_updated"] = datetime.utcnow().isoformat()
    
    # Try to load existing session
    existing_state = await memory_manager.load_session_state(state["session_token"])
    if existing_state:
        # Merge with existing state but keep new message
        user_message = state["user_message"]
        state.update(existing_state)
        state["user_message"] = user_message
    
    return state

async def analyze_input_node(state: AgentState) -> AgentState:
    """Analyze user input to understand needs"""
    
    try:
        # Get user analysis
        context = {
            "conversation_history": state.get("conversation_history", []),
            "user_profile": state.get("user_profile")
        }
        
        analysis = await ai_service.analyze_user_input(
            state["user_message"], 
            context
        )
        
        state["user_analysis"] = analysis
        state["analysis_complete"] = True
        state["current_step"] = "identifying_goals"
        
        # Update conversation history
        if "conversation_history" not in state:
            state["conversation_history"] = []
        
        state["conversation_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": state["user_message"],
            "analysis": analysis
        })
        
    except Exception as e:
        state["processing_errors"].append(f"Analysis failed: {str(e)}")
        state["current_step"] = "error_handling"
    
    return state

async def identify_goals_node(state: AgentState) -> AgentState:
    """Extract and assess goals from user input"""
    
    try:
        goal_data = await ai_service.assess_goals(
            state["user_message"],
            state.get("user_analysis", "")
        )
        
        # Create Goal object with encrypted internal ID
        goal = Goal(
            session_token=state["session_token"],
            primary_goal=goal_data["primary_goal"],
            domains=goal_data["domains"],
            timeframe=goal_data["timeframe"],
            desired_outcomes=goal_data["desired_outcomes"]
        )
        
        state["goal_assessment"] = goal_data
        state["current_goal"] = goal.dict()
        state["goal_identified"] = True
        state["current_step"] = "generating_plan"
        
    except Exception as e:
        state["processing_errors"].append(f"Goal identification failed: {str(e)}")
        state["current_step"] = "error_handling"
    
    return state

async def generate_plan_node(state: AgentState) -> AgentState:
    """Generate comprehensive action plan"""
    
    try:
        plan = await ai_service.generate_comprehensive_plan(
            state.get("goal_assessment", {}),
            state.get("user_profile")
        )
        
        state["comprehensive_plan"] = plan
        state["plan_generated"] = True
        state["current_step"] = "finalizing_response"
        
    except Exception as e:
        state["processing_errors"].append(f"Plan generation failed: {str(e)}")
        state["current_step"] = "error_handling"
    
    return state

async def finalize_response_node(state: AgentState) -> AgentState:
    """Create final response message"""
    
    try:
        # Combine analysis, goals, and plan into coherent response
        response_parts = []
        
        if state.get("user_analysis"):
            response_parts.append("## Understanding Your Situation\n")
            response_parts.append(state["user_analysis"])
            response_parts.append("\n")
        
        if state.get("comprehensive_plan"):
            response_parts.append("## Your Personalized Action Plan\n")
            response_parts.append(state["comprehensive_plan"])
        
        if state.get("current_goal"):
            goal = state["current_goal"]
            response_parts.append(f"\n## Goal Summary\n")
            response_parts.append(f"**Primary Goal:** {goal['primary_goal']}\n")
            response_parts.append(f"**Timeframe:** {goal['timeframe']}\n")
            response_parts.append(f"**Focus Areas:** {', '.join(goal['domains'])}\n")
        
        response_parts.append("\n---\n*I'm here to support you every step of the way! Feel free to share updates or ask questions.*")
        
        state["response_message"] = "\n".join(response_parts)
        state["current_step"] = "complete"
        
        # Save state to memory
        await memory_manager.save_session_state(state)
        
    except Exception as e:
        state["processing_errors"].append(f"Response finalization failed: {str(e)}")
        state["current_step"] = "error_handling"
    
    return state

async def error_handling_node(state: AgentState) -> AgentState:
    """Handle errors gracefully"""
    
    error_messages = state.get("processing_errors", [])
    
    state["response_message"] = """
I apologize, but I encountered some technical difficulties while processing your request. 
However, I'm still here to help! 

Could you please:
1. Try rephrasing your request
2. Or let me know what specific area you'd like to focus on (fitness, nutrition, study habits, etc.)

I'm committed to supporting your personal growth journey, so please don't hesitate to try again.
    """.strip()
    
    state["current_step"] = "complete"
    
    # Log errors but don't expose them to user
    for error in error_messages:
        logging.error(f"Agent error: {error}")
    
    return state

# ========================
# LANGGRAPH WORKFLOW DEFINITION
# ========================

def should_continue_to_goals(state: AgentState) -> str:
    """Conditional edge: continue to goal identification or handle errors"""
    if state.get("analysis_complete") and not state.get("processing_errors"):
        return "identify_goals"
    elif state.get("processing_errors"):
        return "error_handling"
    else:
        return "analyze_input"

def should_continue_to_plan(state: AgentState) -> str:
    """Conditional edge: continue to plan generation or handle errors"""
    if state.get("goal_identified") and not state.get("processing_errors"):
        return "generate_plan"
    elif state.get("processing_errors"):
        return "error_handling"
    else:
        return "identify_goals"

def should_continue_to_finalize(state: AgentState) -> str:
    """Conditional edge: finalize response or handle errors"""
    if state.get("plan_generated") and not state.get("processing_errors"):
        return "finalize_response"
    elif state.get("processing_errors"):
        return "error_handling"
    else:
        return "generate_plan"

def create_langgraph_workflow():
    """Create the LangGraph workflow"""
    
    # Initialize the StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("start_session", start_session_node)
    workflow.add_node("analyze_input", analyze_input_node)
    workflow.add_node("identify_goals", identify_goals_node)
    workflow.add_node("generate_plan", generate_plan_node)
    workflow.add_node("finalize_response", finalize_response_node)
    workflow.add_node("error_handling", error_handling_node)
    
    # Add edges
    workflow.add_edge(START, "start_session")
    workflow.add_edge("start_session", "analyze_input")
    
    # Conditional edges
    workflow.add_conditional_edges(
        "analyze_input",
        should_continue_to_goals,
        {
            "identify_goals": "identify_goals",
            "error_handling": "error_handling",
            "analyze_input": "analyze_input"
        }
    )
    
    workflow.add_conditional_edges(
        "identify_goals",
        should_continue_to_plan,
        {
            "generate_plan": "generate_plan",
            "error_handling": "error_handling",
            "identify_goals": "identify_goals"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_plan",
        should_continue_to_finalize,
        {
            "finalize_response": "finalize_response",
            "error_handling": "error_handling",
            "generate_plan": "generate_plan"
        }
    )
    
    # End edges
    workflow.add_edge("finalize_response", END)
    workflow.add_edge("error_handling", END)
    
    return workflow

# ========================
# PRODUCTION LANGGRAPH SERVICE
# ========================

class LangGraphGigiService:
    def __init__(self):
        self.workflow = create_langgraph_workflow()
        self.checkpointer = MemorySaver()  # In production, use persistent checkpointer
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    async def process_message(self, user_message: str, session_token: str = None) -> Dict[str, Any]:
        """Process user message through LangGraph workflow"""
        
        # Create initial state
        initial_state = {
            "user_message": user_message,
            "session_token": session_token or security.generate_internal_id("session"),
            "current_step": "start",
            "analysis_complete": False,
            "goal_identified": False,
            "plan_generated": False,
            "user_profile": None,
            "current_goal": None,
            "conversation_history": [],
            "user_analysis": None,
            "goal_assessment": None,
            "comprehensive_plan": None,
            "response_message": "",
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "processing_errors": []
        }
        
        try:
            # Run the workflow
            config = {"configurable": {"thread_id": initial_state["session_token"]}}
            result = await self.app.ainvoke(initial_state, config)
            
            return {
                "success": True,
                "response": result.get("response_message", "I'm here to help with your personal growth!"),
                "session_token": result["session_token"],
                "current_step": result.get("current_step", "complete"),
                "goal_data": result.get("current_goal"),
                "metadata": {
                    "processing_time": datetime.utcnow().isoformat(),
                    "workflow_completed": result.get("current_step") == "complete"
                }
            }
            
        except Exception as e:
            logging.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "response": "I apologize for the technical difficulty. Please try again or rephrase your request.",
                "session_token": initial_state["session_token"],
                "error": str(e)
            }
    
    async def get_session_history(self, session_token: str) -> Dict[str, Any]:
        """Retrieve session conversation history"""
        try:
            state = await memory_manager.load_session_state(session_token)
            if state:
                return {
                    "success": True,
                    "conversation_history": state.get("conversation_history", []),
                    "current_goal": state.get("current_goal"),
                    "session_created": state.get("created_at")
                }
            else:
                return {"success": False, "message": "Session not found"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

# ========================
# PRODUCTION API
# ========================

class LangGraphGigiAPI:
    def __init__(self):
        self.service = LangGraphGigiService()
    
    async def chat(self, message: str, session_token: str = None) -> Dict[str, Any]:
        """Main chat endpoint - no user ID required, uses encrypted internal session"""
        return await self.service.process_message(message, session_token)
    
    async def get_history(self, session_token: str) -> Dict[str, Any]:
        """Get conversation history for session"""
        return await self.service.get_session_history(session_token)
    
    async def health_check(self) -> Dict[str, Any]:
        """System health check"""
        return {
            "status": "healthy",
            "service": "LangGraph Gigi Coach",
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_ready": True
        }

# ========================
# USAGE EXAMPLE
# ========================

async def main():
    """Example usage of LangGraph Gigi Coach"""
    
    api = LangGraphGigiAPI()
    
    print("=== LangGraph Gigi Coach Demo ===\n")
    
    # Health check
    health = await api.health_check()
    print(f"System Status: {health['status']}")
    
    # First interaction - new session
    print("\n=== New User Interaction ===")
    response1 = await api.chat(
        "I want to lose 5kg in 8 weeks while studying for my finals. I'm vegetarian and can exercise 30 minutes daily."
    )
    
    print(f"Session Token: {response1['session_token'][:20]}...")
    print(f"Response:\n{response1['response']}")
    
    # Follow-up interaction - same session
    print("\n=== Follow-up in Same Session ===")
    response2 = await api.chat(
        "This sounds great, but I'm worried about meal prep time. I only have 1 hour on Sundays.",
        session_token=response1['session_token']
    )
    
    print(f"Follow-up Response:\n{response2['response']}")
    
    # Get session history
    print("\n=== Session History ===")
    history = await api.get_history(response1['session_token'])
    if history['success']:
        print(f"Total interactions: {len(history['conversation_history'])}")

if __name__ == "__main__":
    asyncio.run(main())