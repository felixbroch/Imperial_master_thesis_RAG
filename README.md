# RAG System for Academic Research

A personal project developed alongside my master's thesis to build a functional and modular Retrieval-Augmented Generation (RAG) system from scratch. This project explores the integration of local language models with external knowledge sources to create more reliable and transparent AI-assisted academic research tools.

## Project Overview

This system was designed to understand and implement the core components of a RAG pipeline, including document indexing, semantic search, and knowledge-grounded text generation. The implementation focuses on academic use cases, specifically supporting queries related to dynamic pricing research and associated technical papers.

### Key Objectives

- Build a complete RAG system using open-source components
- Create transparent document attribution and citation tracking
- Develop modular components for easy extension and customisation

## System Architecture

The RAG system consists of several integrated components:

### 1. Document Processing Pipeline
- **Text Extraction**: Supports multiple formats including PDF, RTF, and plain text
- **Intelligent Chunking**: Section-aware splitting with configurable overlap
- **Metadata Preservation**: Maintains source attribution and citation tracking
- **Citation Extraction**: Automatic identification and linking of academic references

### 2. Embedding and Vector Storage
- **Embedding Model**: BGE-base-en-v1.5 for high-quality semantic representations
- **Vector Database**: FAISS for efficient similarity search and retrieval
- **Indexing Strategy**: L2 distance-based exact nearest neighbour search

### 3. Retrieval System
- **Semantic Search**: Query-based retrieval using embedding similarity
- **Source Attribution**: Maintains links between retrieved chunks and original sources
- **Configurable Retrieval**: Adjustable number of retrieved chunks per query

### 4. Generation Pipeline
- **Local LLM**: Llama.cpp for efficient local inference
- **Structured Prompting**: Academic-focused prompt engineering
- **Citation Handling**: Proper reference formatting and source attribution

## Data Sources

The system currently processes documents related to dynamic pricing research:

### Primary Document
- **Thesis**: "Dynamic Pricing Made Accessible" - A comprehensive study of dynamic pricing implementation in retail

### Reference Papers
- **Chen & Guestrin (2016)**: "XGBoost: A Scalable Tree Boosting System"
- **Fiig et al. (2018)**: "Dynamic Pricing of Airline Offers"
- **Garbarino & Lee (2003)**: "Dynamic Pricing in Internet Retail: Effects on Consumer Trust"
- **Spagnuelo et al. (2017)**: "Metrics for Transparency"

## Installation and Setup

### Installation Steps

1. **Clone the repository**
   ```bash
git clone https://github.com/FelixBROCHIER/Imperial_master_thesis_RAG.git
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the LLM model**
   - Download a quantised model
   - Place the model file in an accessible location
   - Update the model path in the notebook

4. **Prepare documents**
   - Ensure all source documents are in the `RAG_sources_cleaned/` directory
   - Place the thesis document in the `thesis/` directory

### Configuration

The system requires minimal configuration:

1. **Model Path**: Update the LLM model path in the notebook
2. **Document Paths**: Verify document locations match the code
3. **Citation Index**: Modify reference metadata if using different sources