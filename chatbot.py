# chatbot.py 

"""
LogicGuard - Interactive Chatbot Mode
Test your system with any random natural language input.
"""

from logic_validator import LogicValidator
import ollama

def start_chat():
    print("=========================================================")
    print("🤖 LogicGuard Chatbot Active (Type 'exit' to quit)")
    print("=========================================================")
    
    # Initialize your upgraded system
    validator = LogicValidator() 
    
    while True:
        user_input = input("\n🧑 You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
            
        print("🧠 Processing (LogicGuard Analysis + LLM Response)...")
        
        # 1. Run through LogicGuard (Semantic Parser -> Graph)
        validation_result = validator.validate(user_input)
        
        q_type = validation_result.get('template_used')
        if not q_type:
            q_type = 'non-logical'
            
        state = validation_result.get('epistemic_state', 'UNKNOWN')
        
        # 2. Get LLM Answer (STRICT LIMIT TO MAKE IT FAST)
        try:
            response = ollama.chat(
                model='llama3.2:3b',
                messages=[
                    # Yeh line LLM ko lambe essay likhne se rokegi
                    {'role': 'system', 'content': 'You are a helpful AI. Answer in strictly 1 or 2 short sentences only. Be extremely brief.'},
                    {'role': 'user', 'content': user_input}
                ]
            )
            llm_answer = response['message']['content'].strip()
        except Exception as e:
            llm_answer = f"[LLM Error: {e}]"
        
        # 3. Print Results
        print(f"\n🤖 LLM says: {llm_answer}")
        print("-" * 50)
        print(f"🔬 LogicGuard Analysis:")
        print(f"   • Type: {q_type.upper()}")
        print(f"   • Epistemic State: {state}")
        print(f"   • Proof/Reason: {validation_result.get('proof', 'N/A')}")

if __name__ == "__main__":
    start_chat()