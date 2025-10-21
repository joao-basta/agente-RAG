# 🧠 Agente RAG — Retrieval-Augmented Generation for Mental Health Knowledge

This project is an intelligent **RAG (Retrieval-Augmented Generation)** agent capable of answering questions based on a **vector database** populated from a **book on mental illnesses, disorders, and diagnostic approaches**.  

By combining **large language models (LLMs)** with **semantic search** through embeddings, the agent provides contextually grounded and reliable answers from professional reference material rather than relying solely on generative reasoning.

---

## 📘 Project Overview

This repository implements a **Retrieval-Augmented Generation pipeline** that:
1. Ingests and processes a **PDF** containing a comprehensive reference on mental health disorders.  
2. Stores vectorized representations of text segments into a **vector database**.  
3. Retrieves the most relevant passages to the user’s query.  
4. Generates accurate, context-aware responses by combining retrieved knowledge with an LLM.

---

## 🧩 Architecture

```text
User Query
   ↓
Vector Database (embeddings)
   ↓
Retriever → Context Documents
   ↓
LLM (Answer Generation)
   ↓
Final Response
```

**Core Components:**
- **PDF Loader:** Extracts and chunks text from the source book.  
- **Embeddings Generator:** Converts text into numerical vectors for similarity search.  
- **Vector Store:** Enables efficient retrieval of semantically similar chunks.  
- **RAG Pipeline:** Combines retrieved context with an LLM to produce accurate answers.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/joao-basta/agente-RAG.git
cd agente-RAG
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file with your API keys and configuration:
```bash
OPENAI_API_KEY=your_api_key_here
VECTOR_DB_PATH=path_to_your_vector_db
```

---

## 🧠 Usage

### 1. Index the PDF
Before querying, you need to process and store the content:
```bash
python ingest.py
```

### 2. Ask Questions
Once the database is built, start the agent interface:
```bash
python app.py
```
Then, type your question:
```
> What are the diagnostic criteria for bipolar disorder?
```

The agent will retrieve relevant information from the book and return a structured, evidence-grounded answer.

---

## 🧪 Example Queries

- “How do anxiety and depression differ in symptom presentation?”  
- “What are the DSM-5 criteria for schizophrenia?”  
- “How is cognitive behavioral therapy used to treat mood disorders?”

---

## 🧰 Tech Stack

- **Python 3.10+**
- **LangChain / LlamaIndex** (for RAG orchestration)
- **FAISS / Chroma** (vector database)
- **OpenAI / Ollama / Hugging Face models** (LLM integration)
- **dotenv, PyPDF2, tiktoken** (utilities)

---

## 🧑‍⚕️ Ethical Note

This agent is **not a medical diagnostic tool**. It is designed for **educational and research purposes only**.  
All medical or psychological decisions should be made under the guidance of qualified professionals.

---

## 🤝 Contributing

Contributions are welcome!  
If you’d like to improve the retrieval pipeline, model performance, or documentation:

1. Fork the repo  
2. Create a feature branch  
3. Submit a pull request 🚀

---

## 📄 License

This project is released under the **MIT License**.  
See [`LICENSE`](LICENSE) for details.

---

## 👤 Author

**João Basta**  
[GitHub Profile](https://github.com/joao-basta)

---

### 🌟 Support

If you find this project useful, please give it a ⭐ on GitHub to support ongoing improvements!
