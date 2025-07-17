# Setup Guide

This guide provides step-by-step instructions for setting up and running the RAG system for academic research.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: Version 3.10 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 4GB free space for models and data
- **CPU**: Modern multi-core processor (GPU optional but beneficial)

### Software Dependencies
- Python package manager (pip)
- Git for version control
- Jupyter Notebook environment

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Imperial_master_thesis_RAG
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv rag_env

# Activate virtual environment
# On Windows:
rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import sentence_transformers; print('Dependencies installed successfully')"
```

### 4. Download Language Model

#### Option 1: Hugging Face Model Hub
```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download a quantized model (example)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir ./models/
```

#### Option 2: Manual Download
1. Visit a model repository (e.g., TheBloke on Hugging Face)
2. Download a 4-bit quantized model (.gguf format)
3. Place in a `models/` directory in the project root

### 5. Verify Document Structure
Ensure your directory structure matches:
```
Imperial_master_thesis_RAG/
├── rag2.ipynb
├── requirements.txt
├── README.md
├── models/
│   └── your-model.gguf
├── thesis/
│   └── fyp_thesis.txt
└── RAG_sources_cleaned/
    ├── Dynamic_Pricing_in_Internet_Retail_OCR.txt
    ├── Dynamic_Pricing_of_Airline_Offers.txt
    ├── Metrics_for_Transparency.txt
    └── XGBoost_A_Scalable_Tree_Boosting_System copy.txt
```

## Configuration

### 1. Update Model Path
Edit the notebook cell with LLM initialization:
```python
# Update this path to your model location
llm = Llama(
    model_path="./models/your-model.gguf",  # Update path
    n_ctx=4096,
    n_threads=4,
    n_batch=4
)
```

### 2. Adjust System Parameters (Optional)
Modify chunking parameters based on your needs:
```python
# In the chunking function calls
max_chunk_words = 200    # Increase for longer chunks
overlap_words = 40       # Adjust overlap
top_k = 3               # Number of retrieved chunks
```

### 3. Configure Hardware Settings
Adjust based on your system:
```python
# For CPU-only systems
n_threads = 4           # Set to your CPU core count

# For systems with limited RAM
n_batch = 2             # Reduce batch size
```

## Running the System

### 1. Start Jupyter Notebook
```bash
# Navigate to project directory
cd Imperial_master_thesis_RAG

# Start Jupyter
jupyter notebook
```

### 2. Open and Run the Notebook
1. Open `rag2.ipynb` in Jupyter
2. Run cells sequentially (Shift+Enter)
3. Wait for models to load (first run takes longer)

### 3. Test with Sample Queries
Use the provided examples or create your own:
```python
query = "What are the benefits of dynamic pricing in retail?"
prompt, answer = generate_rag_answer(query, model, index, chunk_lookup, citation_index, llm)
print(answer)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors
**Problem**: Out of memory during model loading
**Solution**: 
- Use a smaller quantized model (Q4_0 instead of Q4_K_M)
- Reduce batch size: `n_batch=1`
- Close other applications

#### 2. Slow Performance
**Problem**: Long response times
**Solution**:
- Increase thread count: `n_threads=8`
- Use GPU acceleration if available
- Reduce context window: `n_ctx=2048`

#### 3. Import Errors
**Problem**: Module not found errors
**Solution**:
- Verify virtual environment activation
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Check Python version compatibility

#### 4. Model Loading Issues
**Problem**: Model file not found or corrupted
**Solution**:
- Verify model path in code
- Re-download model file
- Check file permissions

### Performance Optimization

#### For Limited Hardware
```python
# Minimal configuration
llm = Llama(
    model_path="./models/small-model.gguf",
    n_ctx=2048,          # Smaller context
    n_threads=2,         # Fewer threads
    n_batch=1,           # Smaller batch
    n_gpu_layers=0       # CPU only
)
```

#### For High-End Systems
```python
# Optimized configuration
llm = Llama(
    model_path="./models/large-model.gguf",
    n_ctx=4096,          # Larger context
    n_threads=8,         # More threads
    n_batch=8,           # Larger batch
    n_gpu_layers=32      # GPU acceleration
)
```

## Validation

### 1. Test Document Processing
```python
# Verify document loading
print(f"Thesis length: {len(thesis)} characters")
print(f"Total chunks: {len(total_chunks)}")
```

### 2. Test Embedding Generation
```python
# Check embedding dimensions
print(f"Embedding shape: {embeddings.shape}")
print(f"Model loaded: {model}")
```

### 3. Test Retrieval
```python
# Test retrieval with sample query
test_query = "What is dynamic pricing?"
results = retrieve_relevant_chunks(test_query, model, index, chunk_lookup, top_k=3)
print(f"Retrieved {len(results)} chunks")
```

### 4. Test Full Pipeline
```python
# End-to-end test
query = "Explain XGBoost"
prompt, answer = generate_rag_answer(query, model, index, chunk_lookup, citation_index, llm)
print(f"Generated response: {len(answer)} characters")
```

## Advanced Setup

### Custom Document Collections
To use your own documents:

1. **Add Documents**: Place text files in appropriate directories
2. **Update Paths**: Modify file paths in notebook cells
3. **Update Citations**: Modify the citation index dictionary
4. **Rebuild Index**: Run all cells to regenerate embeddings

### Custom Models
To use different models:

1. **Embedding Model**: Replace `SentenceTransformer` model name
2. **LLM Model**: Update model path and parameters
3. **Test Compatibility**: Verify model formats and requirements

### Integration with External Systems
For API integration:

1. **Extract Functions**: Move core functions to separate Python files
2. **Create API Endpoints**: Use Flask or FastAPI
3. **Handle Concurrency**: Implement proper threading/async handling

## Maintenance

### Regular Updates
- Update dependencies: `pip install -r requirements.txt --upgrade`
- Check for model updates
- Validate document integrity

### Backup Strategy
- Save model files and configurations
- Backup processed embeddings
- Version control document changes

### Monitoring
- Track memory usage during operation
- Monitor response quality
- Log performance metrics

## Support

### Documentation
- See `ARCHITECTURE.md` for technical details
- Check `README.md` for project overview
- Review code comments for specific functions

### Common Resources
- Hugging Face documentation for models
- FAISS documentation for vector search
- llama.cpp documentation for LLM inference

### Debugging Tips
1. Enable verbose logging in notebook cells
2. Print intermediate results at each step
3. Test components individually before full pipeline
4. Use small document samples for initial testing
