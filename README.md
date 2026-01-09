<div align="center">
    <img src="_img/GenAI_banner.png" alt="GenAI Overview"/>
</div>

<div align="center">

[**🎯 What is GenAI?**](#1--what-is-genai) |
[**🔧 Types of GenAI**](#2--types-of-generative-ai) |
[**👨‍💻 Builder's Perspective**](#3--builders-perspective) |
[**👤 User's Perspective**](#4--users-perspective) |
[**📚 Projects**](#-projects) |
[**⚡ Installation**](#5--installation)

![Author](https://img.shields.io/badge/Author-mohd--faizy-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-00ADD8?style=for-the-badge&logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-FFD43B?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-0C0D0E?style=for-the-badge&logo=ollama&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT%20Models-412991?style=for-the-badge&logo=openai&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-NLP-orange?style=for-the-badge&logo=huggingface&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>



>This repository provides a practical framework for building Generative AI systems using LangChain for orchestration and HuggingFace for model integration. It focuses on modular, production-grade architectures for complex text generation and multimodal processing. The codebase is actively maintained, with new notebooks and workflow examples added regularly to cover emerging patterns and tools.

<div align="center">
    <img src="_img/GenAI-OverView.png" alt="GenAI Overview" width="800"/>
    <p><em>🔄 Complete Generative AI Pipeline Architecture</em></p>
</div>

---

## 📋 Table of Contents

- [1. What is GenAI?](#1--what-is-genai)
  - [1.1. Core Principles](#11--core-principles)
- [2. Types of Generative AI](#2--types-of-generative-ai)
  - [2.1. Supported Model Types](#21--supported-model-types)
- [3. Builder's Perspective](#3--builders-perspective)
  - [3.1. Foundation Model Architecture](#31--foundation-model-architecture)
  - [3.2. Model Training Pipeline](#32--model-training-pipeline)
  - [3.3. Data Processing](#33--data-processing)
  - [3.4. Model Architecture](#34--model-architecture)
  - [3.5. Training Infrastructure](#35--training-infrastructure)
  - [3.6. Deployment Strategy](#36--deployment-strategy)
  - [3.7. RAG & Orchestration](#37--rag--orchestration)
  - [3.8. Prompts & Evaluation](#38--prompts--evaluation)
- [4. User's Perspective](#4--users-perspective)
  - [4.1. Interface Design](#41--interface-design)
  - [4.2. User Interaction](#42--user-interaction)
  - [4.3. Response Generation](#43--response-generation)
  - [4.4. System Integration](#44--system-integration)
  - [4.5. Performance Metrics](#45--performance-metrics)
  - [4.6. Safety & Ethics](#46--safety--ethics)
- [5. Installation](#5--installation)
  - [5.1. Using UV (Recommended)](#51--using-uv-recommended)
  - [5.2. Alternative Installation](#52--alternative-installation)
- [6. Usage Examples](#6--usage-examples)
- [7. Contributing](#7--contributing)
  - [7.1. Quick Contribution Guide](#71--quick-contribution-guide)
- [8. License](#8--license)
- [9. Credits and Inspiration](#9--credits-and-inspiration)
- [10. Connect with me](#10--connect-with-me)

---



## 1. 🎯 What is GenAI?


> **🧠 Generative AI** is a revolutionary branch of artificial intelligence that creates entirely new content — `text`, `images`, `audio`, `code`, and `video` — by learning intricate patterns and relationships from vast datasets. It doesn't just analyze; it **creates**, **innovates**, and **imagines**.

### 1.1. 🌟 Core Principles

**Generative AI** learns the **distribution of data** to generate new, original samples that maintain the essence of the training data while being completely novel.

<div align="center">

| 🎨 **Domain** | 🔧 **Technology** | 🌟 **Examples** |
|:---:|:---:|:---:|
| **💬 Text** | Large Language Models | ChatGPT, Claude, Gemini |
| **🖼️ Images** | Diffusion Models | DALL-E, Midjourney, Stable Diffusion |
| **💻 Code** | Code Generation LLMs | GitHub Copilot, CodeLlama |
| **🎵 Audio** | Neural Audio Synthesis | ElevenLabs, Mubert |
| **🎬 Video** | Video Generation Models | Sora, RunwayML |

</div>

---

## 2. 🔧 Types of Generative AI

<div align="center">
    <img src="_img/GenAI-Types.png" alt="Types of Generative AI" width="800"/>
    <p><em>🎨 Comprehensive Overview of Generative AI Model Categories</em></p>
</div>

### 2.1. 🎨 Supported Model Types

<details>
<summary><strong>📝 Text Generation Models</strong></summary>

- **🤖 GPT Family**: GPT-3.5, GPT-4, GPT-4 Turbo
- **🔄 T5 Variants**: T5-Small, T5-Base, T5-Large, Flan-T5
- **🧠 BERT Derivatives**: RoBERTa, DeBERTa, ALBERT
- **🦙 Open Source**: Llama 2, Mistral, Falcon

</details>

<details>
<summary><strong>🖼️ Image Generation</strong></summary>

- **🎨 Stable Diffusion**: SD 1.5, SD 2.1, SDXL
- **🎭 DALL-E Integration**: DALL-E 2, DALL-E 3
- **🖌️ Custom Models**: ControlNet, LoRA fine-tuning
- **⚡ Real-time Generation**: LCM, Turbo models

</details>

<details>
<summary><strong>🎵 Audio Processing</strong></summary>

- **🎤 Speech-to-Text**: Whisper, Wav2Vec2
- **🗣️ Text-to-Speech**: Bark, Tortoise TTS
- **🎼 Music Generation**: MusicLM, Jukebox
- **🔊 Audio Enhancement**: Real-ESRGAN Audio

</details>

---

## 3. 👨‍💻 Builder's Perspective

<div align="center">
    <h3>🏗️ Deep Dive into GenAI Architecture</h3>
    <p><em>Understanding the technical foundations that power modern AI systems</em></p>
</div>

### 3.1. 🏗️ Foundation Model Architecture

<div align="center">
    <img src="_img/Builder Perspective/1.1.png" alt="Foundation Model" width="700"/>
    <p><em>Core architectural components of foundation models</em></p>
</div>

**Key Components:**
- **Tokenization Layer**: Converting raw text to numerical representations (BPE, WordPiece).
- **Transformer Blocks**: Self-attention mechanisms (Multi-Head Attention) for context understanding.
- **Embedding Layers**: High-dimensional vector representations of tokens.
- **Output Heads**: Task-specific prediction layers (Causal LM, Sequence Classification).

### 3.2. 🔄 Model Training Pipeline

<div align="center">
    <img src="_img/Builder Perspective/1.2.png" alt="Training Pipeline" width="700"/>
    <p><em>End-to-end model training workflow</em></p>
</div>

**Training Stages:**
- **Pre-training**: Self-supervised learning from massive text corpora (The Pile, CommonCrawl).
- **Fine-tuning**: Supervised Fine-Tuning (SFT) for instruction following.
- **RLHF/DPO**: Aligning models with human preferences using Reinforcement Learning.
- **Evaluation**: Benchmarking on standard datasets (MMLU, GSM8K, HumanEval).

### 3.3. 📊 Data Processing

<div align="center">
    <img src="_img/Builder Perspective/1.3.png" alt="Data Processing" width="700"/>
    <p><em>Data preprocessing and augmentation pipeline</em></p>
</div>

**Processing Steps:**
- **Data Cleaning**: Deduplication, PII redaction, and heuristic filtering.
- **Augmentation**: Synthetic data generation and back-translation.
- **Balancing**: Sampling strategies to ensure dataset diversity.
- **Privacy**: Federated learning and differential privacy techniques.

### 3.4. 🧠 Model Architecture

<div align="center">
    <img src="_img/Builder Perspective/1.4.png" alt="Model Architecture" width="700"/>
    <p><em>Detailed neural network architecture design</em></p>
</div>

**Architecture Elements:**
- **Layer Connections**: Residual connections (ResNet style) and Layer Normalization (Pre-Norm).
- **Activation Functions**: Modern variants like Swish and GeGLU.
- **Positional Embeddings**: RoPE (Rotary Positional Embeddings) or ALiBi for context extension.
- **Optimization**: Flash Attention for efficient computation.

### 3.5. 🖥️ Training Infrastructure

<div align="center">
    <img src="_img/Builder Perspective/1.5.png" alt="Training Infrastructure" width="700"/>
    <p><em>Scalable cloud infrastructure for model training</em></p>
</div>

**Infrastructure Components:**
- **Compute**: H100/A100 Clusters with interconnects (NVLink/InfiniBand).
- **Storage**: High-throughput distributed storage (Lustre, S3).
- **Orchestration**: Kubernetes, Ray, or Slurm for job scheduling.
- **Monitoring**: Weights & Biases or MLflow for experiment tracking.

### 3.6. 🚀 Deployment Strategy

<div align="center">
    <img src="_img/Builder Perspective/1.6.png" alt="Deployment Strategy" width="700"/>
    <p><em>Production deployment and scaling strategies</em></p>
</div>

**Deployment Options:**
- **Model Serving**: vLLM, TGI, or TensorRT-LLM for high-throughput inference.
- **Quantization**: FP8, INT8, or AWQ for reduced memory footprint.
- **Edge AI**: ONNX Runtime or TFLite for mobile deployment.
- **Auto-scaling**: KEDA or Horizontal Pod Autoscaling based on request metrics.

### 3.7. 📚 RAG & Orchestration

**Retrieval-Augmented Generation (RAG)** enhances model accuracy by grounding responses in external data.

- **Vector Databases**: Pinecone, Milvus, Qdrant for semantic search.
- **Chunking Strategies**: Recursive character split vs. semantic chunking.
- **Retrieval Algorithms**: Hybrid search (BM25 + Dense) and Reranking (Cohere, BGE).
- **Orchestration**: LangGraph or LangChain for stateful multi-step workflows.

### 3.8. ⚙️ Prompts & Evaluation

- **Prompt Engineering**: Chain-of-Thought (CoT), ReAct, and System Role definition.
- **Evaluation Frameworks**: RAGAS (Faithfulness, Answer Relevance) and TruLens.
- **LLMOps**: Prompt versioning, trace logging (LangSmith), and dataset management.

---

## 4. 👤 User's Perspective

<div align="center">
    <h3>🎨 Crafting Exceptional User Experiences</h3>
    <p><em>Designing intuitive interfaces for complex AI systems</em></p>
</div>

### 4.1. 🎨 Interface Design

<div align="center">
    <img src="_img/User Perspective/2.1.png" alt="Interface Design" width="700"/>
    <p><em>Modern, intuitive user interface design principles</em></p>
</div>

**Design Principles:**
- **User-Centric**: Clear affordances for AI capabilities vs limitations.
- **Responsive**: Adaptive layouts for mobile, tablet, and desktop.
- **Accessible**: Semantic HTML and ARIA labels for screen readers.
- **Aesthetics**: Clean typography and purposeful whitespace (e.g., Shadcn/UI).

### 4.2. 🤝 User Interaction

<div align="center">
    <img src="_img/User Perspective/2.2.png" alt="User Interaction" width="700"/>
    <p><em>Natural and engaging user interaction patterns</em></p>
</div>

**Interaction Features:**
- **Chat Interface**: Threaded conversations with branching support.
- **Parameter Controls**: Temperature, Top-P, and System Prompt modifiers.
- **Multimodal Input**: Drag-and-drop for images, PDFs, and audio files.
- **Feedback Loops**: Thumbs up/down and text corrections for model improvement.

### 4.3. ⚡ Response Generation

<div align="center">
    <img src="_img/User Perspective/2.3.png" alt="Response Generation" width="700"/>
    <p><em>Lightning-fast response generation pipeline</em></p>
</div>

**Generation Process:**
- **Streaming**: Server-Sent Events (SSE) for perceived latency reduction.
- **Context Awareness**: Sliding window or summarization for long conversations.
- **Customization**: User personas and memory injection.
- **Citations**: Linking generated assertions back to source documents.

### 4.4. 🔗 System Integration

<div align="center">
    <img src="_img/User Perspective/2.4.png" alt="System Integration" width="700"/>
    <p><em>Seamless integration with existing systems</em></p>
</div>

**Integration Capabilities:**
- **API First**: REST/GraphQL endpoints for headless consumption.
- **Webhooks**: Event-driven architecture for async processing.
- **Identity**: OAuth2/OIDC for secure role-based access control (RBAC).
- **Data Persistence**: Postgres (pgvector) or Redis for session storage.

### 4.5. 📈 Performance Metrics

<div align="center">
    <img src="_img/User Perspective/2.5.png" alt="Performance Metrics" width="700"/>
    <p><em>Comprehensive performance monitoring and analytics</em></p>
</div>

**Key Metrics:**
- **TTFT (Time To First Token)**: Optimizing for sub-200ms starts.
- **Tokens/Sec**: Throughput monitoring for cost estimation.
- **Quality**: User acceptance rate and bounce reduction.
- **Error Rates**: Hallucination frequency and fallback triggers.

### 4.6. 🛡️ Safety & Ethics

**Safety Layers** ensure the AI operates within defined boundaries.

- **Guardrails**: Input/Output filtering for PII and toxicity (NeMo Guardrails).
- **Transparency**: Clear labeling of AI-generated content.
- **Bias Mitigation**: System prompts designed to reduce stereotype reinforcement.
- **Rate Limiting**: Preventing abuse and managing cost quotas.

---

## 5. ⚡ Installation

### 5.1. 🐍 Using UV (Recommended)

```bash
# 📥 Clone the repository
git clone https://github.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface.git
cd GenAI-with-Langchain-and-Huggingface

# 🏗️ Initialize UV project
uv init

# 🌐 Create virtual environment
uv venv

# 🔌 Activate environment
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 📦 Install dependencies
uv add -r requirements.txt
```

### 5.2. 🔧 Alternative Installation

<details>
<summary><strong>🐍 Using pip</strong></summary>

```bash
# 📥 Clone repository
git clone https://github.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface.git
cd GenAI-with-Langchain-and-Huggingface

# 🌐 Create virtual environment
python -m venv genai_env

# 🔌 Activate environment
# Linux/Mac:
source genai_env/bin/activate
# Windows:
genai_env\Scripts\activate

# 📦 Install dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>🐍 Using conda</strong></summary>

```bash
# 📥 Clone repository
git clone https://github.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface.git
cd GenAI-with-Langchain-and-Huggingface

# 🌐 Create conda environment
conda create -n genai_env python=3.9
conda activate genai_env

# 📦 Install dependencies
pip install -r requirements.txt
```

</details>

---

## 6. 🛠️ Usage Examples

<details>
<summary><strong>💬 Basic Text Generation</strong></summary>

```python
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 🤖 Initialize model
generator = pipeline("text-generation", 
                    model="microsoft/DialoGPT-medium")
llm = HuggingFacePipeline(pipeline=generator)

# 💬 Generate response
response = llm("Hello, how are you?")
print(response)
```

</details>

<details>
<summary><strong>📄 Document Q&A</strong></summary>

```python
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 📄 Load document
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 🔍 Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# ❓ Setup Q&A chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# 💬 Ask question
answer = qa_chain.run("What is the main topic?")
```

</details>




---

## 7. 🤝 Contributing

### 7.1. 🚀 Quick Contribution Guide

1. **🍴 Fork** the repository
2. **🌿 Create** your feature branch (`git checkout -b feature/AmazingFeature`)
3. **💾 Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **📤 Push** to the branch (`git push origin feature/AmazingFeature`)
5. **🔄 Open** a Pull Request
   
<div align="center">

| 🎯 **Type** | 📝 **Description** | 🔗 **How to Help** |
|:---:|:---:|:---:|
| **🐛 Bug Reports** | Found an issue? | [Open an Issue](https://github.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/issues) |
| **📝 Documentation** | Improve docs | [Edit Documentation](https://github.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/tree/main/docs) |
| **💻 Code** | Add features | [Submit Pull Request](https://github.com/mohd-faizy/GenAI-with-Langchain-and-Huggingface/pulls) |

</div>


---

## 8. 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


---

## 9. 🪙 Credits and Inspiration

This repository draws inspiration from the exceptional educational content developed by Nitish, Krish Naik, and the DataCamp course `Developing LLMs with LangChain`. The implementations and examples provided here are grounded in their comprehensive tutorials on Generative AI, with a particular focus on LangChain and Hugging Face.

## 10. 🔗 Connect with me
<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/F4izy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohd-faizy/)
[![Stack Exchange](https://img.shields.io/badge/Stack_Exchange-1E5397?style=for-the-badge&logo=stack-exchange&logoColor=white)](https://ai.stackexchange.com/users/36737/faizy)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy)

</div>

