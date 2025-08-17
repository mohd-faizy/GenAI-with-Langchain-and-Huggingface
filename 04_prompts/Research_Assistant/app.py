# ============================
# Import Dependencies
# ============================
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

# ============================
# Load Environment Variables
# ============================
load_dotenv()

# ============================
# Initialize the LLM Model
# ============================
model = ChatGroq(
    model="llama-3.1-8b-instant",
    # temperature=0.7,
    # max_tokens=100
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
        "Attention Is All You Need (Vaswani et al., 2017)",
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "GPT-3: Language Models are Few-Shot Learners (Brown et al., 2020)",
        "Diffusion Models Beat GANs on Image Synthesis (Dhariwal & Nichol, 2021)",
        "AlexNet: ImageNet Classification with Deep CNNs (Krizhevsky et al., 2012)",
        "ResNet: Deep Residual Learning for Image Recognition (He et al., 2015)",
        "GANs: Generative Adversarial Nets (Goodfellow et al., 2014)",
        "Word2Vec: Efficient Estimation of Word Representations (Mikolov et al., 2013)",
        "Transformers in Vision: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020, ViT)",
        "DQN: Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)",
        "AlphaGo: Mastering the Game of Go with Deep Neural Networks (Silver et al., 2016)",
        "CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., 2021)"
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

# =================================
# Loading Prompt Template from JSON
# =================================
template = load_prompt(r"E:\01_Github_Repo\GenAI-with-Langchain-and-Huggingface\04_prompts\Research_Assistant\template.json")



# =======================================================================
# Generating Prompt from USER INPUT and then Sending to Model using Chain
# =======================================================================
if st.button('Summarize'):
    chain = template | model          # Create a chain from the template and model
    result = chain.invoke({           # Invoke the chain with user inputs
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.subheader("Summary")
    st.write(result.content)