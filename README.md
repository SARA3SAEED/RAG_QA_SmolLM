# Retrieval Augmented Generation (RAG) Project

This project demonstrates a basic implementation of a **Retrieval Augmented Generation (RAG)** system using several open-source libraries. The goal is to answer questions by retrieving relevant information from a corpus and then using a language model to generate a coherent answer based on the retrieved context.

---

## 📌 Components

- **[Hugging Face Transformers](https://huggingface.co/)**  
  Used for loading and utilizing a pre-trained language model (`SmolLM2-135M-Instruct`) and its tokenizer.

- **[PEFT (Parameter-Efficient Fine-Tuning) with LoRA](https://huggingface.co/docs/peft/index)**  
  Applied to the language model for efficient fine-tuning.

- **[Datasets](https://huggingface.co/docs/datasets/index)**  
  Used to load and process the [`jacobmorrison/truthful_qa-afrikaans-english`](https://huggingface.co/datasets/jacobmorrison/truthful_qa-afrikaans-english) dataset as the corpus for retrieval.

- **[Sentence-Transformers](https://www.sbert.net/)**  
  Used to generate embeddings for the corpus and the user query.

- **[FAISS](https://faiss.ai/)**  
  A library for efficient similarity search, used here to index the corpus embeddings and retrieve relevant questions.

---

## ⚙️ How it Works

1. **Load Model and Tokenizer**  
   Load a pre-trained language model and its tokenizer from the Hugging Face Hub.

2. **Apply LoRA**  
   Apply Parameter-Efficient Fine-Tuning (LoRA) to the language model.

3. **Load and Process Corpus**  
   Load the dataset and extract questions to form the retrieval corpus.

4. **Generate Embeddings**  
   Generate sentence embeddings for each question in the corpus and for the user query.

5. **Build FAISS Index**  
   Index the corpus embeddings using FAISS for fast similarity search.

6. **Retrieve Relevant Questions**  
   Given a user query, retrieve the most similar questions from the corpus using the FAISS index.

7. **Generate Answer**  
   Use the retrieved questions as context for the language model to generate an answer to the user's query.

---

## 🚀 Getting Started


You can open and run the notebook here:

[Open Notebook in Google Colab](
You can open and run the notebook here:

[Open Notebook in Google Colab](https://colab.research.google.com/drive/10O0bNd-a-N4biM3mXuM8XttIjzrFqKhz?usp=sharing)
To run this notebook, install the required libraries:

```bash
pip install transformers peft datasets sentence-transformers faiss-cpu
