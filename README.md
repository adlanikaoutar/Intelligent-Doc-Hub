# Intelligent Doc Hub

LLM-powered multi-format RAG platform for intelligent document search, contextual Q&A, multi-document comparison, and automated report generation.

## Overview

Intelligent Doc Hub is a Retrieval-Augmented Generation (RAG) platform designed to process and interact with documents in multiple formats, including PDF, Word, Excel, CSV, and PNG.

The platform combines document processing, semantic search, vector indexing, and Large Language Models (LLMs) to provide contextual answers based on the user's documents.

## Features

- Multi-format document ingestion
- Text and data extraction
- Document preprocessing and chunking
- Embedding generation
- Vector indexing with FAISS
- Semantic document search
- Natural language Q&A with an LLM
- Contextual answer generation
- Multi-document comparison
- Automated report generation
- Search by identifier and keywords

## RAG Pipeline

```text
Documents
    ↓
Ingestion
    ↓
Extraction
    ↓
Preprocessing
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Vector Index
    ↓
Semantic Retrieval
    ↓
Relevant Context
    ↓
LLM
    ↓
Contextual Answer
