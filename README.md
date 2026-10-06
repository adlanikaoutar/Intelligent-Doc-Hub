# Intelligent Doc Hub

> LLM-powered multi-format RAG platform for intelligent document search, contextual Q&A, document comparison, and information synthesis.

##  Overview

**Intelligent Doc Hub** is a local Retrieval-Augmented Generation (RAG) platform developed with Python and LangChain.

The application allows users to load and interact with documents through natural language. It combines document processing, semantic search, vector embeddings, FAISS, and a local Large Language Model (LLM) to retrieve relevant information and generate contextual responses.

The platform currently supports **PDF, DOCX, TXT/MD, and Excel files**.

---

##  Main Features

###  1. Q&A Classique — RAG

Users can ask questions about the indexed documents using natural language.

The system:

- Retrieves the most relevant document chunks
- Uses semantic similarity search
- Provides the retrieved context to the LLM
- Generates a contextual answer in French
- Displays the source documents used for the response
- Reduces hallucinations through a strict context-based prompt

---

###  2. Document Comparison

The application provides a document comparison feature.

Users can:

- Select an existing document from the database
- Upload a temporary document for comparison
- Retrieve the most similar document chunks
- Exclude the reference document from the comparison
- Generate an automatic comparison report

The generated report contains:

- **Points de Similarité**
- **Divergences / Différences**

---

###  3. Q&A Intelligent par Information Clé

This module combines semantic retrieval with exact text matching.

Users can search for an:

- Identifier
- Reference number
- Name
- Keyword
- Specific piece of information

The system retrieves relevant document chunks and generates a French synthesis containing:

1. Relevant documents
2. Summary of the information found

---

###  4. Dynamic Document Indexing

New documents can be uploaded directly from the Streamlit interface.

The application:

1. Saves the uploaded document
2. Loads the document using the appropriate LangChain loader
3. Splits the content into chunks
4. Generates embeddings
5. Adds the new chunks to the FAISS vector store
6. Saves the updated index

---

##  Supported Formats

| Format | Support |
|--------|---------|
| PDF | ✅ |
| DOCX | ✅ |
| TXT | ✅ |
| MD | ✅ |
| XLS | ✅ |
| XLSX | ✅ |
| CSV | ❌ |
| PNG | ❌ |

> CSV and PNG are not currently implemented in the document loaders. They can be added in future versions.

---

##  RAG Architecture

```text
                    ┌─────────────────────┐
                    │      Documents      │
                    │ PDF / DOCX / TXT    │
                    │ MD / XLS / XLSX     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Document Ingestion  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Extraction &        │
                    │ Preprocessing       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Recursive Character │
                    │ Text Splitter       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SentenceTransformer │
                    │ Embeddings          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FAISS Vector DB   │
                    └──────────┬──────────┘
                               │
                         User Query
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Similarity Search   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Relevant Context    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Mistral 7B          │
                    │ via Ollama          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Contextual Answer   │
                    └─────────────────────┘
