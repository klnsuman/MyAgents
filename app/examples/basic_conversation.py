from app.agents.basic_agent import BasicAgent

def main():
    """Run a basic conversation with the agent."""
    # Create the agent
    agent = BasicAgent()
    
    # Add a system message
    agent.add_system_message("You are a helpful AI assistant that specializes in explaining complex topics simply.")
    
    # Have a conversation
    questions = [
        "What is an AI agent?",
        "How do AI agents differ from regular LLMs?",
        "Can you explain the concept of agent memory?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = agent.chat(question)
        print(f"Agent: {response}")

if __name__ == "__main__":
    main()
