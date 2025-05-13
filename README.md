# 📚 MCQ Creator App

Generate **Multiple Choice Questions (MCQs)** from your **PDF documents** using the power of **LangChain**, **Hugging Face Transformers**, and **Pinecone** vector search! This app turns your boring ol' PDFs into spicy quizzes in seconds 🌶️💡

---

## 🛠️ Tech Stack

- 🧠 **LangChain** – to build the question-answering chain  
- 🪄 **Hugging Face Transformers** – for embeddings & text generation  
- 📁 **PyPDFLoader** – to extract text from PDFs  
- 🧩 **Recursive Text Splitter** – for manageable chunking  
- 📦 **Pinecone Vector DB** – to store and search semantic chunks  
- 🌍 **Dotenv** – to manage your API keys safely  

---

## 🚀 Features

- 📄 Load PDF documents  
- ✂️ Smartly split long text into smaller chunks  
- 🔍 Embed and store in Pinecone  
- 🤖 Ask questions, get answers using Mistral-7B  
- 📘 Convert answers into structured **MCQs**  
- 🎯 Output in a clean JSON-like structure  

---

## 📂 Project Structure

```

MCQ-creator-app/
│
├── doc/
│   └── document.pdf          # Your input PDF
│
├── app.py                   # Main app script
├── .env                     # Environment variables (API keys)
└── README.md                # You're reading this fabulous file

````

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
````

Or install them manually:

```bash
pip install langchain langchain-community langchain-huggingface langchain-pinecone pypdf python-dotenv pinecone-client sentence-transformers
```

---

## 🔐 Environment Setup

Create a `.env` file in your project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
PINECONE_API_KEY=your_pinecone_api_key
```

---

## ⚙️ How to Use

1. Place your target PDF inside the `doc/` folder.
2. Run the app:

```bash
python app.py
```

It will:

* Load and chunk the PDF
* Embed it and upload to Pinecone
* Answer your question
* Generate a structured MCQ from the answer

---

## 🧪 Sample Output

```json
{
  "question": "What kind of token is an NFT?",
  "choices": "Fungible token, Non-fungible token, Utility token, Governance token",
  "answer": "Non-fungible token"
}
```
