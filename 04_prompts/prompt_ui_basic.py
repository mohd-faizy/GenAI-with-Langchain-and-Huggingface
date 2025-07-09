# ============================
# Import Dependencies
# ============================
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

# ============================
# Load Environment Variables
# ============================
load_dotenv()

# ============================
# Initialize the LLM Model
# ============================
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=100
)

# ============================
# Streamlit UI Setup
# ============================
st.title("AI Research Paper Summarizer")
st.subheader("Generate tailored summaries of foundational AI research papers")

# ---------- User Input Controls ----------
paper_input = st.selectbox(
    "Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# ============================
# Prompt Template Definition
# ============================
template = PromptTemplate(
    input_variables=['paper_input', 'style_input', 'length_input'],
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the selected style and length.
""",
    validate_template=True
)

# ============================
# Generate Prompt from Inputs
# ============================
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

# ============================
# Generate and Display Output
# ============================
if st.button('Summarize'):
    result = model.invoke(prompt)
    st.subheader("Summary")
    st.write(result.content)