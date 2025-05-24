# LangChain Components

![author](https://img.shields.io/badge/author-mohd--faizy-red)

# Langchain components

![image.png](https://raw.githubusercontent.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/refs/heads/main/_img/_langCompIMG/Lang_comp.png)

# 🔹 **1. Models – The Core of LangChain**

---

### 📌 **What Are LangChain Models?**

- 🧠 The **Models component** is the **core interface** to interact with AI models (LLMs & Embedding Models).
- 🔄 LangChain is **model-agnostic** – you can switch between different LLM providers with minimal code changes.
- 🛠️ Solves the **standardization problem** – every provider (`OpenAI`, `Gemini`, `Anthropic`, etc.) has different APIs, but LangChain offers one unified interface.


### 📚 **Why Are Models Important?**

- ✅ Most important component of LangChain – it's where the AI “thinks.”
- 🤖 Handles both **language generation** (chatbots, agents) and **vector embedding** (search, retrieval).
- 🏗️ Acts as a **foundation** for the other 5 components: Prompts, Chains, Memory, Indexes, Agents.


### 🔍 **Challenges Solved by LangChain Models**

1. 🧱 **Huge Size** of LLMs (100GB+) → Solved via API access.
2. 🔌 **Different APIs for Different Providers** → LangChain unifies them.
3. 🔁 **No Standardized Output/Input Handling** → LangChain parses and handles it uniformly.


### 🤹‍♂️ **Types of Models in LangChain**

1. 🗣️ **Language Models (LLMs)**
    - Input: Text
    - Output: Text
    - Use cases: Chatbots, summarization, translation, coding.
    - Providers: `OpenAI`, `Claude`, `Hugging Face`, `Bedrock`, `Mistral`, `Vertex AI`, `Azure`.
  
2. 🧭 **Embedding Models**
    - Input: Text
    - Output: Vector (numerical representation)
    - Use case: `Semantic Search` / `Vector DB`
    - Providers: `OpenAI`, `Mistral` AI, `IBM`, `Llama`, etc.


### 🧪 **Features Supported Across Models**

- 🧰 Tool calling
- 📦 JSON / Structured output
- 🧑‍💻 Local execution
- 📸 Multimodal input (e.g., images + text)


## 💡 **Code Examples for LangChain Models**

### 1️⃣ Load a Chat Model (OpenAI)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)
response = llm.invoke("What is the capital of France?")

print(response.content)

```


### 2️⃣ Load a Chat Model (Anthropic Claude)

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-opus-20240229")
response = llm.invoke("Explain quantum entanglement in simple terms.")

print(response.content)

```


### 3️⃣ Load an Embedding Model (OpenAI)

```python
from langchain_openai import OpenAIEmbeddings

embedder = OpenAIEmbeddings()
vector = embedder.embed_query("What is machine learning?")

print(vector[:5])  # Print first 5 values of the vector

```


### 4️⃣ Switch Between Providers with 1 Line

```python
# Switching from OpenAI to Mistral (Minimal change)
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-small")
response = llm.invoke("Summarize the plot of Inception.")

print(response.content)

```


### 5️⃣ Use Local Language Model (e.g., Llama.cpp or Ollama)

```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama2")
response = llm.invoke("What are black holes?")

print(response.content)

```


### 6️⃣ Advanced: JSON Output from a Chat Model

```python
llm = ChatOpenAI(model="gpt-4", temperature=0.2)
response = llm.invoke("Return a JSON of 3 countries and their capitals.")

# Ensure the output is structured JSON (using function calling or prompt formatting)
print(response.content)  # Should be a structured response like: {"France": "Paris", ...}

```


### ✅ **Summary**

- The **Models component** provides a **standardized, pluggable way** to interact with any LLM or embedding model.
- Enables rapid experimentation and development with **minimal vendor lock-in**.
- Supports both **language tasks (text in → text out)** and **vector embeddings (text in → vector out)**.


---
![divider.png](https://raw.githubusercontent.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/refs/heads/main/_img/_langCompIMG/divider.png)

---

# 🔹**2. Prompts – Crafting the Right Questions for LLMs**

### 📌 **What Are Prompts in LangChain?**

- ✍️ A **prompt** is the **input or instruction** you give to an LLM.
- ❗ The **quality of the output** depends directly on the **quality of the prompt**.
- 🧠 Even small changes in the prompt can **hugely change** the LLM’s response.
- 🧪 Example:
  - "Explain Linear Regression in an academic tone"
  - vs
  - "Explain Linear Regression in a fun tone"

        🔄 → Two completely different outputs!

### 🎓 **Why Are Prompts Important?**

- 💥 Prompts are the **most sensitive and influential** part of working with LLMs.
- 🧑‍🔬 The rise of **Prompt Engineering** as a field (and job!) proves how central prompts are.
- 🧩 LangChain provides a **Prompts component** to manage, customize, and structure prompts efficiently.


### 🧰 **What the Prompts Component Offers**

- 🔄 **Dynamic prompts** – insert values at runtime using placeholders.
- 🧑‍⚖️ **Role-based prompts** – guide the LLM to take on a persona or expertise.
- 🧪 **Few-shot prompts** – train the model by showing it examples of the behavior you expect.
- 📦 Reusability – create prompt **templates** you can use again and again in different contexts.


## 🧠 **Types of Prompts in LangChain**

### 1️⃣ Dynamic & Reusable Prompts

- 🔧 Use placeholders like `{topic}` or `{tone}` that get filled dynamically.
- ✅ Example: `"Summarize this {topic} in a {tone} tone."`

### 2️⃣ Role-Based Prompts

- 🧑‍⚕️ Use a system-level prompt like: `"You are an experienced doctor."`
- 👤 Then ask: `"Explain symptoms of viral fever."`

### 3️⃣ Few-Shot Prompts

- 🎓 Give **input-output examples** to teach the model before the real query.
- 📊 Example: Show how messages map to categories before asking it to classify a new one.


## 🧪 **Code Examples for LangChain Prompts**

### 1️⃣ Basic Prompt Template

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is {topic}?")
formatted = prompt.format(topic="Quantum Computing")
print(formatted)

```


### 2️⃣ Dynamic Multi-Variable Prompt

```python
prompt = PromptTemplate.from_template("Summarize the topic '{topic}' in a {tone} tone.")
print(prompt.format(topic="Climate Change", tone="fun"))

```


### 3️⃣ Role-Based Prompt with System Message

```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are an expert {profession}."),
    HumanMessagePromptTemplate.from_template("Tell me about {topic}.")
])

formatted = prompt.format_messages(profession="architect", topic="modern skyscrapers")
print([msg.content for msg in formatted])

```


### 4️⃣ Few-Shot Prompt Template

```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"message": "I can't log in", "category": "Technical Issue"},
    {"message": "Please update my billing info", "category": "Billing"}
]

example_template = PromptTemplate.from_template("User: {message}\nCategory: {category}\n")
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Classify the user's query:",
    suffix="User: {query}\nCategory:",
    input_variables=["query"]
)

print(few_shot_prompt.format(query="How do I change my credit card info?"))

```


### 5️⃣ Combine Prompt with Chat Model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke(prompt.format(topic="Machine Learning", tone="formal"))
print(response.content)

```


### 6️⃣ Prompt with Custom Jinja Template (Advanced)

```python
from langchain.prompts import PromptTemplate

template = """You are a {role}.
Answer the question below clearly and professionally.

Question: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["role", "question"])
print(prompt.format(role="data scientist", question="What is overfitting?"))

```


### ✅ **Summary**

- The **Prompts component** gives **full control** over how you talk to LLMs.
- Enables **reusable, flexible, and structured** prompt design.
- Makes your apps **more reliable and intelligent** by controlling how LLMs interpret input.
- ✨ Essential for building smart, adaptive, and role-aware AI apps.

---
![divider.png](https://raw.githubusercontent.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/refs/heads/main/_img/_langCompIMG/divider.png)


# 🔹 **3. Chains – Build Smart Pipelines for LLM Workflows**

### 📌 **What Are Chains in LangChain?**

- 🔗 Chains are used to **link multiple steps** of an LLM app into a **single automated pipeline**.
- 🤖 LangChain is **named after Chains** – that’s how fundamental they are!
- ⚙️ They let you build **sequential**, **parallel**, or **conditional** flows between components like LLMs, tools, and memory.


### ⚡ **Why Use Chains?**

- 🔄 Automatically passes the **output of one step as the input** to the next.
- 🧼 Avoids repetitive manual code to handle data transfer between steps.
- 🚀 Lets you design **multi-step AI applications** that work as one smooth pipeline.


### 🛠️ **Real-World Use Case (Sequential Chain Example)**

### 🔁 English Text ➡️ Hindi Translation ➡️ Hindi Summary

1. Step 1: Translate English to Hindi (LLM 1)
2. Step 2: Summarize Hindi text (LLM 2)

    ✅ Chains handle this flow without manual intervention — just input English text and get the final Hindi summary.


### 🔍 **Types of Chains in LangChain**

### 1️⃣ **Sequential Chains**

- Steps run **one after another** in order.
- Example: Translate → Summarize → Format → Output

### 2️⃣ **Parallel Chains**

- 🧠 Run multiple LLMs **simultaneously** and combine results.
- Example: Same input sent to 3 LLMs to generate different takes → Combine in final report.

### 3️⃣ **Conditional Chains**

- 🤔 Branching logic: behavior changes based on input/response.
- Example: If user feedback is negative → Send alert to support; else → Send thank-you note.


## 💻 **Code Examples for LangChain Chains**


### 1️⃣ Basic LLMChain (1-step flow)

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate.from_template("Translate the following English text to Hindi:\n\n{text}")

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("I love learning about AI.")
print(result)

```


### 2️⃣ SequentialChain: Translation ➡️ Summarization

```python
from langchain.chains import SequentialChain

translate_prompt = PromptTemplate.from_template("Translate to Hindi:\n\n{text}")
summary_prompt = PromptTemplate.from_template("Summarize this Hindi text in under 100 words:\n\n{text}")

translate_chain = LLMChain(llm=llm, prompt=translate_prompt, output_key="translated")
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, input_key="translated")

full_chain = SequentialChain(
    chains=[translate_chain, summary_chain],
    input_variables=["text"],
    output_variables=["translated", "text"]
)

result = full_chain.run({"text": "Artificial Intelligence is transforming the world."})
print(result)

```


### 3️⃣ Simple Conditional Chain (If-Else Logic)

```python
from langchain.chains import TransformChain

def route_feedback(inputs):
    feedback = inputs["feedback"]
    return {"action": "thank user" if "good" in feedback.lower() else "alert support"}

router = TransformChain(input_variables=["feedback"], output_variables=["action"], transform=route_feedback)

result = router.run({"feedback": "The product is not working."})
print(result)

```


### 4️⃣ Parallel Chain (Mock Conceptual Example)

```python
from langchain.chains import SimpleSequentialChain

prompt1 = PromptTemplate.from_template("Write a poem about {topic}")
prompt2 = PromptTemplate.from_template("Write a joke about {topic}")

chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# In practice, you could run these chains in parallel using asyncio or LangGraph (experimental)
poem = chain1.run("robots")
joke = chain2.run("robots")

print("Poem:\n", poem)
print("Joke:\n", joke)

```


### 5️⃣ Chain with Memory Integration (Preview)

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
prompt = PromptTemplate.from_template("You are a chatbot. User said: {input}")

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
print(chain.run("Hello, how are you?"))
print(chain.run("What did I just say?"))

```


### ✅ **Summary**

- 🧩 **Chains** simplify **multi-step workflows** in LLM applications.
- 💡 They abstract away manual code and let you focus on logic and flow.
- 🧠 Types:
  - **Sequential** – one step after another
  - **Parallel** – multiple steps at once
  - **Conditional** – smart branching
- 📦 Combine Chains with Prompts, Memory, and Agents for powerful apps.

---
![divider.png](https://raw.githubusercontent.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/refs/heads/main/_img/_langCompIMG/divider.png)

---

# 🧠 **4. Memory – Remembering Past Conversations in LangChain**

### 📌 **What Is Memory in LangChain?**

- 🔁 Most LLMs like GPT are **stateless** — they forget everything after each message.
- ❌ If you ask:
  - "Who is Narendra Modi?"
  - Then: "How old is he?"
  - → The model doesn’t remember who *"he"* is.
- 🧠 **Memory solves this problem** by maintaining **context across turns** in a conversation.



### 🚀 **Why Is Memory Important?**

- 🗣️ Makes **chatbots and assistants feel natural and human-like**.
- 🧾 Keeps track of what users say — no need to repeat questions.
- 🤖 Essential for building **stateful AI applications** like customer service bots, AI tutors, assistants, etc.


### 🔍 **Types of Memory in LangChain**

| 🧠 Type | 📋 Description | 💡 Use Case |
| --- | --- | --- |
| **ConversationBufferMemory** | Stores **full chat history** | Best for short conversations |
| **ConversationBufferWindowMemory** | Stores **last N messages** | Great for recent context without overloading |
| **ConversationSummaryMemory** | Stores a **summary of conversation** | Ideal for long chats, saves cost |
| **Custom Memory** | Store **special facts or variables** | Good for personalized assistants |



## 💻 **Code Examples for LangChain Memory**


### 1️⃣ Basic Memory Integration with `LLMChain`

```python
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory()

prompt = PromptTemplate.from_template("You are a helpful bot. User said: {input}")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Simulate conversation
print(chain.run("Who is Virat Kohli?"))
print(chain.run("What team does he play for?"))  # Remembers previous message

```


### 2️⃣ Using `ConversationBufferWindowMemory`

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)  # Remembers last 2 interactions

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
print(chain.run("Explain Machine Learning."))
print(chain.run("Give an example."))
print(chain.run("What did I just say?"))  # Only remembers 2 last messages

```


### 3️⃣ Using `ConversationSummaryMemory` (with summarization)

```python
from langchain.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(llm=llm)  # Uses LLM to auto-summarize past chat

chain = LLMChain(llm=llm, prompt=prompt, memory=summary_memory)
print(chain.run("Explain the plot of Inception."))
print(chain.run("Who was the main character?"))  # Will have access to summary, not full text

```


### 4️⃣ Custom Memory Example (storing variables)

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "My name is Alex"}, {"output": "Nice to meet you, Alex!"})
memory.save_context({"input": "I live in Delhi"}, {"output": "Delhi is a great city."})

# Access stored memory
print(memory.load_memory_variables({}))  # Returns entire conversation history

```


### 5️⃣ Use Memory with a ChatPromptTemplate

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain

chat_prompt = ChatPromptTemplate.from_template("Human: {input}\nAI:")

conversation = ConversationChain(
    llm=llm,
    prompt=chat_prompt,
    memory=ConversationBufferMemory()
)

print(conversation.run("Tell me a joke."))
print(conversation.run("Another one please!"))  # Keeps previous context

```


### ✅ **Summary**

- 🧠 **Memory makes LangChain apps stateful** — just like real conversations.
- 💬 It keeps track of what was said earlier and gives the model **context**.
- 🔧 Use different memory types based on your app’s need:
  - Full history (Buffer)
  - Recent messages only (Window)
  - Summarized context (Summary)
- 🧩 Combine Memory with Chains, Prompts, and Models to build **real conversational agents**.

---
![divider.png](https://raw.githubusercontent.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/refs/heads/main/_img/_langCompIMG/divider.png)


# 🗂️ **5. Indexes – Letting LLMs Use Your Private Data**

### 🤔 **Why Do We Need Indexes?**

- 🤖 LLMs like ChatGPT **do not know your private data**.
  - ❌ "What’s the leave policy of XYZ company?" → Can't answer.
  - Because it's **not in the training data**.
- ✅ We solve this using **Indexes** in LangChain to:
  - **Connect LLMs to external data** (e.g. PDFs, websites).
  - **Search and retrieve only what’s needed** from this data.
  - Use it for **answering questions** based on it.


### 🧱 **The 4 Core Sub-Components of Indexes**

| 🔢 | 🔧 Component | 📋 Role |
| --- | --- | --- |
| 1️⃣ | **Document Loader** | Loads your file (PDF, CSV, Notion, Drive, etc.) |
| 2️⃣ | **Text Splitter** | Breaks large text into smaller chunks |
| 3️⃣ | **Vector Store** | Stores chunk embeddings for similarity search |
| 4️⃣ | **Retriever** | Finds the best chunks for a user query |


### 📊 **How It Works (Simplified Flow)**

```
PDF file (RulesBook.pdf)
     ↓
[1] Document Loader ➜ Load the document
     ↓
[2] Text Splitter ➜ Split into small chunks
     ↓
[3] Vector Store ➜ Embed chunks + Store
     ↓
[4] Retriever ➜ Semantic search on user query
     ↓
     LLM (answers based on relevant chunks)

```



## 💻 **LangChain Indexes – Code Example**


### 🧾 1. Load Your PDF Document

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("XYZ_Company_Policy.pdf")
docs = loader.load()

```


### ✂️ 2. Split Into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

```


### 🧠 3. Create Embeddings + Vector Store

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

```


### 🔍 4. Setup Retriever and Ask Questions

```python
retriever = vector_store.as_retriever()

query = "What is the official leave policy?"
relevant_docs = retriever.get_relevant_documents(query)

for doc in relevant_docs:
    print(doc.page_content)

```


### ✅ Optional: Use with RetrievalQA Chain

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

print(qa_chain.run("What is the resignation notice period?"))

```


## 🧠 **Why Indexes Are Crucial**

- 🔓 They unlock the **power of private, local, or custom data**.
- ⚙️ They work seamlessly with other components (like Prompts + Chains).
- 📚 Perfect for building:
  - Internal company chatbots
  - Personalized tutors
  - Research assistants
  - FAQ bots on your data


## 🔁 Recap of the 4 Sub-Components

| 📁 DocumentLoader | 🔢 TextSplitter | 🧠 VectorStore | 🕵️‍♂️ Retriever |
| --- | --- | --- | --- |
| Load files | Split into chunks | Store vectors | Find relevant chunks |

---
![divider.png](https://raw.githubusercontent.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/refs/heads/main/_img/_langCompIMG/divider.png)


# 🤖 **6. Agents – The Smartest, Action-Oriented Component**

### 💡 **What Are Agents?**

- Think of Agents as **chatbots with superpowers** ⚡
- They don’t just *respond* – they can **think, decide, and act**
- Agents combine:
  - 🧠 Reasoning (`"What should I do first?"`)
  - 🔧 Tool use (`"Let me use a calculator or weather API"`)
  - 🧩 Integration with other LangChain components


### 🎯 **How Are Agents Different from Chatbots?**

| Chatbot | Agent |
| --- | --- |
| Just responds to queries | **Performs actions** |
| Can’t use APIs or tools | **Can call tools, APIs, functions** |
| Gives answers | **Finds answers + performs real tasks** |


### 🧠 **Two Key Superpowers of Agents**

1. **🧩 Reasoning**: They break problems down into logical steps.
    - Often via *Chain of Thought* prompting.
2. **🔧 Tool Use**: They can use:
    - 🔢 Calculator
    - 🌦️ Weather API
    - 🔍 Search
    - 📅 Calendar
    - 📊 Custom APIs
    - 💾 Local Indexes
    - and more...


### 🔁 **How an Agent Works (Behind-the-Scenes Flow)**

1. User: *“Multiply today’s temperature in Delhi by 3”*
2. Agent thinks:
    - “I need to find Delhi’s temperature”
    - “Then multiply it by 3”
3. Agent uses 🔧 weather tool → gets 25°C
4. Agent uses 🔧 calculator tool: 25 × 3 = 75
5. Agent returns: **“The result is 75”**


### 💡 **10 Awesome Real-World Agent Examples**

| 🔢 # | 🌍 Real-World Use Case | 🧠 Tools / Steps |
| --- | --- | --- |
| 1️⃣ | "Convert today’s INR to USD" | Currency API + calculator |
| 2️⃣ | "Remind me to call mom at 7 PM" | Calendar API |
| 3️⃣ | "Summarize this YouTube transcript and email me" | Summarizer + Email API |
| 4️⃣ | "Book a cab from home to airport" | Location API + Cab Booking API |
| 5️⃣ | "Give me weather in 3 cities and compare them" | Weather tool × 3 + comparison logic |
| 6️⃣ | "Tell me the latest stock price of Apple and calculate 5% profit on 10 shares" | Stock API + calculator |
| 7️⃣ | "Find top 3 tourist places in Japan and translate to Hindi" | Search + Translator tool |
| 8️⃣ | "Fetch leave balance from HR system and show calendar view" | HR API + calendar integration |
| 9️⃣ | "Search PDF for 'termination policy' and translate to Marathi" | PDF retriever + translator |
| 🔟 | "Ask a question, retrieve from my docs, and save the result to Notion" | Vector index + Notion API |


### 🔧 **Minimal Code Example: Agent with Tools**

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun, Calculator

# Define Tools
search = DuckDuckGoSearchRun()
calc = Calculator()

tools = [
    Tool(name="Search", func=search.run, description="Useful for web search"),
    Tool(name="Calculator", func=calc.run, description="Useful for math calculations")
]

# Load model
llm = ChatOpenAI(temperature=0)

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask a smart question
agent.run("What's the population of Japan divided by 3?")

```


```python
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun, Calculator

search = DuckDuckGoSearchRun()
calc = Calculator()

llm = ChatOpenAI(temperature=0)

tools = [
    Tool(name="Search", func=search.run, description="Web search"),
    Tool(name="Calculator", func=calc.run, description="Math ops")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### 🧩 **How Agents Connect to Other Components**

| 🧠 Component | 🤝 Role |
| --- | --- |
| **Models** | For reasoning + responses |
| **Prompts** | Guide agent thinking |
| **Chains** | Internal pipelines agent may call |
| **Memory** | Track long conversations or tasks |
| **Indexes** | Retrieve knowledge to reason on |

Agents are the **glue** that orchestrates all the above when needed.


### 📌 **Summary**

- 📦 **Agents = Intelligent Orchestrators**
- 🔍 Understand what needs to be done
- 🛠️ Use tools and APIs to **act**
- 🔗 Leverage all other LangChain components
- 🚀 Power behind **real-world, useful LLM apps**

---
---

![divider.png](https://raw.githubusercontent.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/refs/heads/main/_img/_langCompIMG/divider.png)

---
---

# ✅ Commonly Used Built-in Tools in LangChain

These are prebuilt and can be used directly or extended:

### 📚 1. `WikipediaQueryRun`

- Search and summarize Wikipedia content.

```python
from langchain.tools import WikipediaQueryRun
```

---

### 🔍 2. `DuckDuckGoSearchRun`

- Performs web searches using DuckDuckGo.

```python
from langchain.tools import DuckDuckGoSearchRun
```

---

### 🧮 3. `Calculator`

- Evaluates mathematical expressions using Python.

```python
from langchain.tools import Calculator
```

---

### 📆 4. `LLMMathTool`

- Parses and evaluates math problems with reasoning using the LLM + calculator.

---

### 📂 5. `PythonREPLTool`

- Runs Python code interactively inside a REPL.

```python
from langchain.tools.python.tool import PythonREPLTool
```

---

### 🌐 6. `SerpAPIWrapper`

- Uses the Google Search API (SerpAPI) for rich search queries.

```python
from langchain.tools import SerpAPIWrapper
```

---

### 💬 7. `HumanInputRun`

- Asks for manual input from a human (useful in CLI tools).

---

### 🧾 8. `TerminalTool`

- Allows executing real commands in a shell (use cautiously).

---

### 🔐 9. `RequestsGetTool`, `RequestsPostTool`

- Send HTTP GET or POST requests.

```python
from langchain.tools.requests.tool import RequestsGetTool
```

---

### 🧠 10. `RetrievalQA` or `VectorStoreQATool`

- Allows agents to query a **Vector Store** (e.g., FAISS, Pinecone) via semantic search.

---

## 🧩 Custom Tools

You can define **any function as a tool**:

```python
from langchain.tools import tool

@tool
def get_greeting(name: str) -> str:
    return f"Hello, {name}!"
```

Then include it in the `tools` list when initializing an agent.

---

## 💡 Specialized Integrations / Tools via LangChain Plugins

LangChain also provides wrappers for:

| Service/API | Tool Type |
| --- | --- |
| Wolfram Alpha | `WolframAlphaQueryRun` |
| Google Search (Serp) | `SerpAPIWrapper` |
| OpenWeatherMap | Custom API + `RequestsGetTool` |
| SQL Databases | `SQLDatabaseToolkit` |
| Python Code Execution | `PythonREPLTool` or `LLMMathTool` |
| Notion API | Custom Tool using SDK |
| Zapier API | `ZapierNLARunAction` |
| File I/O Tools | Read/write files from local system (custom) |

---

## 🔧 How Agent Uses Tools

When you pass tools to the agent:

```python
agent = initialize_agent(
    tools=[search, calc, custom_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

```

LangChain parses the prompt, detects tool requirements, and **automatically selects and invokes tools** in real-time.

---

## 🧠 Summary: Categories of Tools

| Category | Tools |
| --- | --- |
| **Math** | `Calculator`, `LLMMathTool`  |
| **Search** | `DuckDuckGoSearchRun`, `SerpAPIWrapper`, `Wikipedia` |
| **Code** | `PythonREPLTool`, `TerminalTool` |
| **Web Requests** | `RequestsGetTool`, `RequestsPostTool`  |
| **Memory / Data** | `VectorStoreQATool`, `RetrievalQA`, `RedisTool`  |
| **APIs** | `Wolfram`, `Notion`, `Zapier`, `OpenWeather`  |
| **Human Input** | `HumanInputRun` |

![divider.png](https://raw.githubusercontent.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/refs/heads/main/_img/_langCompIMG/divider.png)

