from typing import TypedDict, Dict, Any
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Initialize memory (stores last 5 exchanges)
memory = ConversationBufferWindowMemory(k=5, return_messages=True)
from langchain.chat_models import ChatOpenAI

# Initialize the GROQ LLM using OpenAI-compatible endpoint
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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",  # replace with your GROQ API key
    model="llama3-70b-8192",      # or another supported model like mixtral
)
# Define agent prompts
AGENT_PROMPTS = {
    "billing": """You are an expert billing support agent. Specialize in:
- Payment processing
- Invoice disputes
- Refund requests
- Subscription management""",
    
    "tech": """You are a senior technical support engineer. Specialize in:
- Login/authentication issues
- System errors
- Performance problems
- Technical troubleshooting""",
    
    "shipping": """You are a shipping logistics specialist. Specialize in:
- Order tracking
- Delivery issues
- Shipping methods
- Package location"""
}

class State(TypedDict):
    user_problem: str
    specialist_agent: str
    specialist_response: str
    response: str
    memory: Dict[str, Any]

# Problem Classification with memory
def problem_classification(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        """Previous conversation: {history}
        
        Classify this query:
        - 'billing' for payments/invoices
        - 'tech' for technical issues
        - 'shipping' for deliveries
        
        Query: {user_problem}
        Classification:"""
    )
    
    chain = prompt | llm
    classification = chain.invoke({
        "user_problem": state["user_problem"],
        "history": state["memory"].get("history", "No previous conversation")
    }).content
    
    return {"specialist_agent": classification.strip()}

# Specialist agent template with agent_prompt
def create_specialist_prompt(agent_type: str):
    return ChatPromptTemplate.from_template(
        """Agent Specialization: {agent_prompt}
        
        Conversation history: {history}
        
        Current issue: {user_problem}
        
        As a {agent_type} specialist, provide detailed assistance:"""
    )

def billing_agent(state: State) -> State:
    prompt = create_specialist_prompt("billing")
    chain = prompt | llm
    response = chain.invoke({
        "agent_prompt": AGENT_PROMPTS["billing"],
        "user_problem": state["user_problem"],
        "history": state["memory"].get("history", "No previous conversation"),
        "agent_type": "billing"
    }).content
    return {"specialist_response": response}

def tech_agent(state: State) -> State:
    prompt = create_specialist_prompt("tech")
    chain = prompt | llm
    response = chain.invoke({
        "agent_prompt": AGENT_PROMPTS["tech"],
        "user_problem": state["user_problem"],
        "history": state["memory"].get("history", "No previous conversation"),
        "agent_type": "technical"
    }).content
    return {"specialist_response": response}

def shipping_agent(state: State) -> State:
    prompt = create_specialist_prompt("shipping")
    chain = prompt | llm
    response = chain.invoke({
        "agent_prompt": AGENT_PROMPTS["shipping"],
        "user_problem": state["user_problem"],
        "history": state["memory"].get("history", "No previous conversation"),
        "agent_type": "shipping"
    }).content
    return {"specialist_response": response}

# Summary agent with agent context
def summary_agent(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        """Agent Specialization Context: {agent_prompt}
        
        Full conversation: {history}
        
        Original problem: {user_problem}
        Specialist solution: {specialist_response}
        
        Craft a final response that:
        1. References the specialist's expertise
        2. Incorporates the solution
        3. Provides clear next steps"""
    )
    
    agent_prompt = AGENT_PROMPTS.get(state["specialist_agent"].lower(), "")
    
    chain = prompt | llm
    final_response = chain.invoke({
        "agent_prompt": agent_prompt,
        "user_problem": state["user_problem"],
        "specialist_response": state["specialist_response"],
        "history": state["memory"].get("history", "No previous conversation")
    }).content
    
    # Update memory
    memory.save_context(
        {"input": state["user_problem"]},
        {"output": final_response}
    )
    
    return {
        "response": final_response,
        "memory": memory.load_memory_variables({})
    }

# Routing logic
def route_agent(state: State) -> str:
    agent = state["specialist_agent"].lower()
    if 'billing' in agent:
        return 'billing_agent'
    elif 'tech' in agent:
        return 'tech_agent'
    elif 'shipping' in agent:
        return 'shipping_agent'
    return "tech_agent"  # default fallback

# Build the workflow
workflow = StateGraph(State)
workflow.add_node('classify', problem_classification)
workflow.add_node('billing_agent', billing_agent)
workflow.add_node('tech_agent', tech_agent)
workflow.add_node('shipping_agent', shipping_agent)
workflow.add_node('summary_agent', summary_agent)

workflow.set_entry_point('classify')
workflow.add_conditional_edges(
    'classify',
    route_agent,
    {
        "billing_agent": "billing_agent",
        "tech_agent": "tech_agent",
        "shipping_agent": "shipping_agent"
    }
)
workflow.add_edge('billing_agent', 'summary_agent')
workflow.add_edge('tech_agent', 'summary_agent')
workflow.add_edge('shipping_agent', 'summary_agent')
workflow.add_edge('summary_agent', END)

app = workflow.compile()

def run_conversation(query: str):
    # Load current memory
    current_memory = memory.load_memory_variables({})
    
    # Execute workflow
    results = app.invoke({
        "user_problem": query,
        "memory": current_memory
    })
    
    return results

# Test the conversation
print("=== First Query ===")
response1 = run_conversation("I was charged twice for my subscription")
print(response1['response'])

print("\n=== Follow-up ===")
response2 = run_conversation("Can you confirm the refund was processed?")
print(response2['response'])

print("\n=== Memory ===")
print(memory.load_memory_variables({}))