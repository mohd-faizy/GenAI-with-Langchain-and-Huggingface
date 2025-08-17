# -------------------------------
# Imports
# -------------------------------
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import json
import os

# Load environment variables (for API keys etc.)
load_dotenv()


# -------------------------------
# 1. Define the LLM
# -------------------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile"   # You can also set temperature, max_tokens here
)


# -------------------------------
# 2. Define the Chat Template
# -------------------------------
# - System message: tells the AI its role
# - MessagesPlaceholder: stores past conversation
# - Human message: the new user query
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])


# -------------------------------
# 3. Load Chat History (from JSON file)
# -------------------------------
if os.path.exists("chat_history.json"):
    # If file exists, load past conversation
    with open("chat_history.json", "r") as f:
        data = json.load(f)

    # Convert JSON dicts into LangChain message objects
    chat_history = []
    for item in data:
        if item["type"] == "system":
            chat_history.append(SystemMessage(content=item["content"]))
        elif item["type"] == "human":
            chat_history.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            chat_history.append(AIMessage(content=item["content"]))
else:
    # If no history file, start fresh with a system message
    chat_history = [SystemMessage(content="You are a helpful AI Assistant!")]


# -------------------------------
# 4. Chat Loop (interactive)
# -------------------------------
while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user input to chat history
    chat_history.append(HumanMessage(content=user_input))

    # Build prompt using template (history + new query)
    prompt = chat_template.invoke({
        "chat_history": chat_history,
        "query": user_input
    })

    # Get AI response from model
    response = model.invoke(prompt)

    # Add AI response to chat history
    chat_history.append(AIMessage(content=response.content))

    # Show response to user
    print("AI:", response.content)


# -------------------------------
# 5. Save Chat History (to JSON)
# -------------------------------
data = []

for msg in chat_history:
    if isinstance(msg, SystemMessage):  # isinstance(object, ClassName) checks if something is of a specific type (or subclass).
        data.append({"type": "system", "content": msg.content})
    elif isinstance(msg, HumanMessage):
        data.append({"type": "human", "content": msg.content})
    elif isinstance(msg, AIMessage):
        data.append({"type": "ai", "content": msg.content})

with open("chat_history.json", "w") as f:
    json.dump(data, f, indent=2)

print("\n✅ Chat history saved to chat_history.json")
