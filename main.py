import asyncio
import sys
import os
from datetime import datetime

# Import the agent from your existing file
from core import LangGraphGigiAPI

class TerminalGigiAgent:
    def __init__(self):
        """Initialize the terminal-based agent"""
        self.api = LangGraphGigiAPI()
        self.session_token = None
        self.conversation_count = 0
        
    def print_welcome(self):
        """Display welcome message"""
        print("=" * 50)
        print("🤖 GIGI - Your Personal Growth Coach")
        print("=" * 50)
        print("Hi! I'm Gigi, ready to help you achieve your goals!")
        print("Type your message and press Enter to chat.")
        print("Commands: 'exit' to quit, 'history' to see conversation history")
        print("=" * 50)

    async def get_conversation_history(self):
        """Retrieve and display conversation history"""
        if not self.session_token:
            print("No conversation history yet.")
            return
            
        history = await self.api.get_history(self.session_token)
        if history['success']:
            print(f"\n📊 Conversation Summary (Total: {len(history['conversation_history'])} interactions)")
            print("-" * 40)
            
            for i, convo in enumerate(history['conversation_history'][-5:], 1):  # Last 5 interactions
                timestamp = convo.get('timestamp', 'Unknown time')
                message = convo.get('user_message', 'No message')[:60]
                print(f"{i}. {timestamp}: {message}...")
                
            if history.get('current_goal'):
                goal = history['current_goal']
                print(f"\n🎯 Current Goal: {goal['primary_goal']}")
                print(f"⏰ Timeframe: {goal['timeframe']}")
        else:
            print("Could not retrieve conversation history.")

    async def process_user_input(self, user_input: str):
        """Process user input and get AI response"""
        try:
            # Show thinking indicator
            print("🤔 Gigi is thinking...")
            
            # Call the agent
            response = await self.api.chat(
                message=user_input,
                session_token=self.session_token
            )
            
            if response['success']:
                # Update session token for continuity
                self.session_token = response.get('session_token')
                self.conversation_count += 1
                
                return response['response']
            else:
                return f"Sorry, I encountered an error: {response.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Sorry, I encountered a technical issue: {str(e)}"

    async def run(self):
        """Main terminal interaction loop"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                print(f"\n[Message #{self.conversation_count + 1}]")
                user_input = input("You: ").strip()
                
                # Handle special commands
                if user_input.lower() == 'exit':
                    print("\n👋 Thanks for chatting with Gigi! Take care!")
                    if self.conversation_count > 0:
                        print(f"Session summary: {self.conversation_count} messages exchanged")
                    break
                
                elif user_input.lower() == 'history':
                    await self.get_conversation_history()
                    continue
                    
                elif not user_input:
                    print("Please enter a message or type 'exit' to quit.")
                    continue
                
                # Process the message with AI agent
                response = await self.process_user_input(user_input)
                
                # Display response
                print("\n" + "=" * 50)
                print("🤖 Gigi:")
                print("=" * 50)
                print(response)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Session interrupted by user.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again or type 'exit' to quit.")

def main():
    """Entry point for the terminal agent"""
    # Set up Windows event loop policy if needed
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Create and run the terminal agent
    agent = TerminalGigiAgent()
    
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nAgent terminated by user.")
    except Exception as e:
        print(f"Error starting agent: {e}")

if __name__ == "__main__":
    main()
