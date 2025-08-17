from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),      # include chat history in prompt before user query for context
    ('human', '{query}')
])

chat_history = []

# load chat history Text
# with open('chat_history.txt') as f:
#     chat_history.extend(f.readlines())

# load chat history JSON
with open('chat_history.json', 'r') as f:
    chat_history = json.load(f)

print(chat_history)

# create prompt
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund?'
})

print(prompt)