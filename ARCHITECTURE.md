# System Architecture Documentation

## Overview

This document provides a detailed technical overview of the RAG (Retrieval-Augmented Generation) system implemented for academic research assistance. The system is designed to process academic documents, extract semantic information, and generate contextually relevant responses with proper source attribution.

## Architecture Components

### 1. Document Processing Layer

#### Text Extraction
- **Supported Formats**: Plain text (.txt), PDF (.pdf), RTF (.rtf)
- **Preprocessing**: Whitespace normalisation and section detection

#### Chunking Strategy
The system implements a sophisticated chunking approach:

```python
def split_by_section_headings_with_meta(text, source_type, max_chunk_words=200, overlap_words=40)
```

**Key Features:**
- Section-aware splitting using regex patterns for numbered headings
- Sliding window approach with configurable overlap
- Metadata preservation for each chunk
- Citation extraction and tracking

### 2. Embedding and Vector Storage

#### Embedding Model
- **Model**: BAAI/bge-base-en-v1.5
- **Dimensions**: 768
- **Optimisation**: English academic text
- **Batch Processing**: Efficient encoding of multiple chunks

#### Vector Database
- **Engine**: FAISS (Facebook AI Similarity Search)
- **Index Type**: IndexFlatL2 (L2 distance)
- **Search Method**: Exact nearest neighbour
- **Data Type**: float32 for compatibility

### 3. Retrieval System

#### Semantic Search
The retrieval system implements semantic search using:

**Process Flow:**
1. Query encoding using the same embedding model
2. Vector similarity search in FAISS index
3. Metadata retrieval using index mapping
4. Relevance ranking based on distance scores

#### Source Attribution
Each retrieved chunk maintains:
- Original document source
- Section context
- Citation information
- Relevance score

### 4. Response Generation

#### Prompt Engineering
The system uses structured prompts with:

**Prompt Structure:**
- Role definition for academic assistance
- Source-attributed context
- Clear citation instructions
- Query specification

#### LLM Integration
- **Engine**: llama.cpp for local inference (Mistral 4bit running locally in this example)
- **Model Format**: GGUF (quantized weights)
- **Parameters**: Configurable temperature, top_p, max_tokens
- **Context Window**: 4096 tokens

### 5. Citation System

#### Citation Extraction
Automatic identification of academic citations using regex:

**Pattern Matching:**
- Format: `(Author, Year)` or `(Author et al., Year)`
- Validation against citation index
- Support for multiple citations per chunk

#### Citation Index
Structured metadata for each reference:
```python
{
    "citation": "(Chen and Guestrin, 2016)",
    "title": "XGBoost: A Scalable Tree Boosting System",
    "link": "https://dl.acm.org/doi/10.1145/2939672.2939785"
}
```

## Data Flow

### 1. Preprocessing Pipeline
```
Documents → Text Extraction → Section Detection → Chunking → Metadata Attachment
```

### 2. Indexing Pipeline
```
Chunks → Embedding Generation → Vector Index Creation → Metadata Mapping
```

### 3. Query Pipeline
```
User Query → Query Embedding → Similarity Search → Context Formatting → LLM Generation
```