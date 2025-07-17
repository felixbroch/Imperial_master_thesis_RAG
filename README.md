# RAG System for Academic Research

A personal project developed alongside my master's thesis to build a functional and modular Retrieval-Augmented Generation (RAG) system from scratch. This project explores the integration of local language models with external knowledge sources to create more reliable and transparent AI-assisted academic research tools.

## Project Overview

This system was designed to understand and implement the core components of a RAG pipeline, including document indexing, semantic search, and knowledge-grounded text generation. The implementation focuses on academic use cases, specifically supporting queries related to dynamic pricing research and associated technical papers.

### Key Objectives

- Build a complete RAG system using open-source components
- Implement local inference for privacy and control
- Create transparent document attribution and citation tracking
- Develop modular components for easy extension and customisation
- Support academic research workflows with proper source attribution

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

### Prerequisites
- Python 3.10 or higher
- At least 8GB RAM for model loading
- 4GB free disk space for models and data

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Imperial_master_thesis_RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the LLM model**
   - Download a quantized model
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

## Usage

### Basic RAG Query
```python
# Initialize the system
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
llm = Llama(model_path="path/to/model.gguf", n_ctx=4096)

# Generate answer
prompt, answer = generate_rag_answer(
    query="What are the benefits of dynamic pricing?",
    embed_model=model,
    index=index,
    chunk_lookup=chunk_lookup,
    citation_index=citation_index,
    llm=llm
)
```

### Running the Jupyter Notebook
1. Start Jupyter: `jupyter notebook`
2. Open `rag2.ipynb`
3. Run cells sequentially to build the system
4. Modify queries in the final cells to test different questions

### Customisation Options

**Chunk Size**: Adjust `max_chunk_words` for different context lengths
**Overlap**: Modify `overlap_words` for context preservation
**Retrieval Count**: Change `top_k` in retrieval functions
**LLM Parameters**: Adjust `temperature`, `top_p`, and `max_tokens`

## System Components

### Document Chunking
The system implements intelligent chunking that:
- Preserves document structure using section headers
- Maintains overlapping context between chunks
- Tracks source attribution and citations
- Handles multiple document types uniformly

### Embedding Generation
Uses BGE-base-en-v1.5 for creating dense vector representations:
- High-quality semantic understanding
- Optimised for English academic text
- Efficient batch processing capabilities

### Vector Search
FAISS provides efficient similarity search:
- Exact nearest neighbour search using L2 distance
- Scalable to large document collections
- Fast query processing for real-time applications

### Response Generation
Local LLM inference ensures:
- Privacy-preserving processing
- Consistent model behaviour
- Customisable generation parameters
- Transparent source attribution

## Known Limitations

### Model Dependencies
- Requires significant computational resources for local inference
- Model quality depends on the chosen LLM weights
- Limited by the embedding model's language understanding

### Document Processing
- Currently optimised for academic text structure
- Manual citation index maintenance required
- Limited file format support

### Scalability
- Memory requirements scale with document collection size
- Vector index rebuilding required for new documents
- Sequential processing of large document sets

## Future Extensions

### Enhanced Retrieval
- Hybrid search combining semantic and keyword matching
- Dynamic chunk sizing based on content complexity
- Multi-modal support for figures and tables

### Advanced Generation
- Citation verification and validation
- Multi-document synthesis capabilities
- Structured output formatting (tables, lists, references)

### User Interface
- Web-based interface for non-technical users
- Real-time query processing
- Document upload and management system

### Integration Options
- API endpoints for external applications
- Database backends for persistent storage
- Cloud deployment configurations

## Contributing

This is a personal learning project, but suggestions and improvements are welcome. Areas of particular interest include:

- Improved document processing pipelines
- Enhanced citation extraction and validation
- Better prompt engineering for academic contexts
- Performance optimisations for large document collections

## Acknowledgements

This project was developed as part of my master's thesis research under the supervision of Professor Pierre Pinson at Imperial College London. The work explores the intersection of dynamic pricing research and modern AI technologies, with particular focus on transparency and accessibility in academic research tools.

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

- Fiig, T., Le Guen, R., & Gauchet, M. (2018). Dynamic pricing of airline offers. IATA White Paper.

- Garbarino, E., & Lee, O. F. (2003). Dynamic pricing in Internet retail: Effects on consumer trust. Psychology & Marketing, 20(6), 495-513.

- Spagnuelo, D., Bartolini, C., & Lenzini, G. (2017). Metrics for transparency. In Data Protection and Privacy: (In)visibilities and Infrastructures (pp. 1-21).

## Licence

This project is intended for educational and research purposes. Please ensure compliance with the licences of all included models and libraries when using this system.
