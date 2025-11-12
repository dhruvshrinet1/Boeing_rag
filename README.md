# Boeing 737 Manual RAG Service

A Retrieval-Augmented Generation (RAG) API service for querying the Boeing 737 technical manual with accurate page citations.

## Overview

This service provides a REST API endpoint that processes questions about the Boeing 737 manual and returns grounded answers with specific page references. It uses Google Gemini models for embeddings and generation, with a hybrid retrieval system combining dense vector search (FAISS) and sparse keyword search (BM25) for optimal accuracy.

## Features

- **Accurate Page Citations**: Returns 1-based page numbers from the source PDF
- **Hybrid Retrieval**: Combines FAISS vector similarity with BM25 keyword search
- **Re-ranking**: Uses cross-encoder re-ranking for improved relevance
- **Structured Responses**: JSON output with separate answer and page fields
- **Production Ready**: FastAPI with proper error handling and logging
- **Modular Architecture**: Separate services for document processing, retrieval, and generation

## Architecture

The solution consists of three main components:

### 1. Document Processing Service (`src/document_processor.py`)
- **PDFProcessor**: Extracts text from PDF with page-level metadata tracking
- **VectorStoreManager**: Creates and manages FAISS vector store
- Each chunk maintains its source page number (1-based index)
- Configurable chunk size and overlap for optimal retrieval

### 2. RAG Service (`src/rag_service.py`)
- **HybridBoeingRAGService**: Enhanced RAG with BM25 + Dense + Re-ranking
- Extracts unique page numbers from retrieved documents
- Provides detailed retrieval information for evaluation

### 3. API Service (`src/api.py`)
- FastAPI application with `/query` endpoint
- Returns JSON: `{"answer": "...", "pages": [1, 5, 12]}`
- Health check and retrieval info endpoints

## Installation

### Prerequisites
- Python 3.11+
- Google Gemini API key

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Assignment "
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

4. **Place the Boeing manual PDF**
Ensure the PDF is at: `data/document_analysis/Boeing B737 Manual-1.pdf`

## Usage

### Starting the Server

Run the server using:
```bash
python main.py
```

The server will start at `http://0.0.0.0:8000`

### API Endpoints

#### Query Endpoint
**POST** `/query`

Request:
```json
{
  "question": "What is the fuel capacity of the Boeing 737?"
}
```

Response:
```json
{
  "answer": "The Boeing 737 has a fuel capacity of approximately 6,875 U.S. gallons (26,035 liters) in its standard configuration...",
  "pages": [42, 43, 87]
}
```

#### Health Check
**GET** `/health`

Returns service status.

#### Retrieval Info (Debug)
**GET** `/retrieval-info?question=your+question`

Returns detailed retrieval information including scores and document previews.

### Example Usage

```bash
# Using curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the emergency procedures for engine failure?"}'

# Using Python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is the maximum takeoff weight?"}
)
print(response.json())
```

## Solution Design

### How It Works

1. **Document Indexing** (First Run Only)
   - PDF is processed page-by-page using PyMuPDF
   - Each page is split into chunks (1000 chars with 200 overlap)
   - Chunks are embedded using Google's `text-embedding-004`
   - FAISS index is created and saved to disk
   - Page metadata is preserved in each chunk

2. **Query Processing**
   - User question is embedded using the same model
   - Hybrid retrieval:
     - Dense: FAISS similarity search (top 10)
     - Sparse: BM25 keyword search (top 15)
     - Ensemble: Reciprocal rank fusion
   - Re-ranking: Cross-encoder re-ranks results (final top 8)
   - Page extraction: Unique page numbers collected from retrieved chunks
   - Generation: Gemini 2.0 Flash generates answer from context

3. **Response Formation**
   - Answer text from LLM
   - Sorted unique page numbers (1-based index from PDF)
   - Returned as JSON

### Key Design Decisions

#### 1. Page-Level Metadata Tracking
**Challenge**: Ensure accurate page citations in responses.

**Solution**:
- Extract pages individually before chunking
- Attach page number (1-based) to each chunk's metadata
- Preserve metadata through entire pipeline
- Extract unique pages from retrieved chunks

**Why**: This guarantees that page numbers refer to actual PDF pages, not document-printed page numbers.

#### 2. Hybrid Retrieval System
**Challenge**: Balance between semantic understanding and keyword matching.

**Solution**:
- FAISS for semantic similarity
- BM25 for keyword/term matching
- Ensemble fusion (50-50 weights)
- Cross-encoder re-ranking for final selection

**Why**:
- Technical manuals contain specific terminology (BM25 excels)
- Conceptual questions need semantic understanding (FAISS excels)
- Hybrid approach covers both cases
- Re-ranking improves precision of final results

#### 3. Chunk Size Selection
**Challenge**: Balance between context and precision.

**Solution**:
- Chunk size: 1000 characters
- Overlap: 200 characters

**Why**:
- 1000 chars: Enough context for meaningful passages
- 200 overlap: Prevents information loss at boundaries
- Smaller chunks = more precise page citations
- Tested and optimized for technical content

#### 4. Model Selection
**Choice**: Google Gemini models

**Rationale**:
- **Embeddings** (`text-embedding-004`): High quality, 768 dimensions
- **LLM** (`gemini-2.0-flash`): Fast, accurate, good at following instructions
- Temperature: 0 for consistent, factual responses
- API key provided in requirements

### Challenges Faced and Solutions

#### Challenge 1: Page Number Accuracy
**Problem**: Initial approach lost page information during chunking.

**Solution**:
- Modified document processing to extract pages first
- Attached metadata before chunking
- Verified page numbers match PDF indices

#### Challenge 2: Retrieval Quality
**Problem**: Pure vector search missed specific technical terms.

**Solution**:
- Implemented hybrid retrieval (BM25 + Dense)
- Added cross-encoder re-ranking
- Tuned retrieval parameters (k values)

#### Challenge 3: Context Window Management
**Problem**: Too many/large chunks exceeded context limits.

**Solution**:
- Limited final retrievals to top 8 chunks
- Formatted context with clear page markers
- Optimized chunk sizes

#### Challenge 4: Response Grounding
**Problem**: Ensuring answers stay within context.

**Solution**:
- Crafted specific system prompts
- Set temperature to 0
- Instructed LLM to only use provided context
- Structured context with page references

## Project Structure

```
Assignment/
├── main.py                 # Entry point - run server
├── requirements.txt        # Dependencies
├── .env.example           # Environment variables template
├── .env                   # Actual environment variables (gitignored)
├── config/
│   └── config.yaml        # Model and retrieval configuration
├── data/
│   └── document_analysis/
│       └── Boeing B737 Manual-1.pdf  # Source document
├── src/
│   ├── document_processor.py    # PDF processing & vector store
│   ├── rag_service.py           # RAG logic & retrieval
│   ├── api.py                   # FastAPI application
│   ├── document_analyser/       # Legacy document handlers
│   └── document_chat/           # Legacy chat components
├── model/
│   └── models.py          # Pydantic models & enums
├── prompt/
│   └── prompt_library.py  # Prompt templates
├── utils/
│   ├── model_loader.py    # LLM & embedding loaders
│   └── config_loader.py   # YAML config loader
├── logger/
│   └── custom_logging.py  # Structured logging
├── exception/
│   └── custom_exception.py  # Custom exceptions
├── faiss_index/           # FAISS vector store (generated)
└── logs/                  # Application logs
```

## Configuration

### Environment Variables

See `.env.example` for all available options:
- `GOOGLE_API_KEY`: Your Google Gemini API key (required)
- `BOEING_MANUAL_PATH`: Path to PDF file
- `FAISS_INDEX_DIR`: Directory for vector store
- `USE_HYBRID_RETRIEVAL`: Enable hybrid retrieval (true/false)
- `TOP_K`: Number of documents to retrieve
- `API_HOST`, `API_PORT`: Server configuration

### Model Configuration

Edit `config/config.yaml` to change:
- Embedding model
- LLM model and parameters
- Retrieval top_k value

## Testing

### Manual Testing
```bash
# Start server
python main.py

# In another terminal, test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the fuel system components?"}'
```

### Automated Testing
```bash
pytest test.py
```

## Performance Optimization

- **First run**: ~30-60 seconds (PDF processing + indexing)
- **Subsequent runs**: <1 second startup (loads existing index)
- **Query latency**: ~2-4 seconds (retrieval + generation)
- **Memory**: ~500MB (FAISS index + models)

## Logging

Structured JSON logs are written to `logs/` directory with:
- Request/response details
- Retrieval information
- Error traces
- Performance metrics

## Future Enhancements

1. **Multi-document support**: Index multiple manuals
2. **Caching**: Cache common queries
3. **Batch processing**: Handle multiple questions
4. **Fine-tuned embeddings**: Train on aviation domain
5. **Answer confidence scoring**: Return confidence metrics
6. **Streaming responses**: Stream LLM output

## Troubleshooting

### Issue: "No existing vector store found"
**Solution**: Ensure PDF is at correct path and run once to create index.

### Issue: "API key error"
**Solution**: Check `.env` file has valid `GOOGLE_API_KEY`.

### Issue: "Import errors"
**Solution**: Run `pip install -r requirements.txt` again.

### Issue: "Empty page numbers returned"
**Solution**: Check PDF has text (not just images). Use OCR if needed.

## License

This project was developed for the Boeing RAG challenge. All rights to the developed service are retained by the developer as specified in the challenge requirements.

## Contact

For questions or issues, please create an issue in the GitHub repository.