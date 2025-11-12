# Boeing 737 Manual RAG Service

RAG-based Q&A system for Boeing 737 technical manual with accurate page citations.

## What it does

Ask questions about the Boeing 737 manual, get answers with page references:

```bash
POST /query
{
  "question": "What is the fuel capacity?"
}

Response:
{
  "answer": "The Boeing 737 has a fuel capacity of approximately 6,875 U.S. gallons...",
  "pages": [42, 43]
}
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your Google API key to `.env`:
```bash
GOOGLE_API_KEY=your_key_here
```

3. Place PDF at `data/document_analysis/Boeing B737 Manual-1.pdf`

4. Run:
```bash
python main.py
```

Server starts at `http://localhost:8000`

## How it works

### Indexing (first run only)
- Extracts text from PDF page-by-page
- Splits into chunks (1000 chars, 200 overlap)
- Creates FAISS vector index with embeddings
- Saves to disk for reuse

### Query processing
1. **Hybrid retrieval**: BM25 (keyword) + FAISS (semantic)
2. **Reranking**: Cross-encoder sorts by relevance
3. **Generation**: Gemini generates answer from top chunks
4. **Page extraction**: Collects unique page numbers from sources

## Tech stack

- **Embeddings**: Google `text-embedding-004`
- **LLM**: Google Gemini 2.5 Pro
- **Vector DB**: FAISS
- **Keyword search**: BM25
- **Reranker**: Cross-encoder
- **API**: FastAPI
- **PDF parsing**: PyMuPDF

## Project structure

```
├── main.py                    # Start server
├── src/
│   ├── api.py                 # FastAPI endpoints
│   ├── rag_service.py         # Hybrid retrieval + reranking
│   ├── document_processor.py  # PDF processing
│   └── query_expander.py      # Query expansion with LLM
├── prompt/prompt_library.py   # All prompts
├── model/models.py            # Pydantic schemas
├── utils/
│   ├── model_loader.py        # Load LLM & embeddings
│   └── config_loader.py       # YAML config
├── config/config.yaml         # Model settings
└── faiss_index/               # Generated vector store
```

## Configuration

Edit `.env` for:
- `GOOGLE_API_KEY` - Required
- `PROCESS_IMAGES` - Enable multimodal image processing (default: false)
- `TOP_K` - Retrieval count (default: 20)
- `RERANK_TOP_K` - Final docs after reranking (default: 10)

Edit `config/config.yaml` to change models or parameters.

## Image processing (optional)

Enable multimodal image extraction:

```bash
# In .env
PROCESS_IMAGES=true
```

This uses Gemini to describe images/diagrams in the manual. **Warning**: Slower and uses more API calls.

## API endpoints

### `POST /query`
Main endpoint for questions.

**Request:**
```json
{"question": "What are the emergency procedures?"}
```

**Response:**
```json
{
  "answer": "...",
  "pages": [12, 15, 18]
}
```

### `GET /health`
Check if service is ready.

### `GET /`
API info.

## Evaluation

Run evaluation with retrieval scoring:

```bash
python run_evaluation.py
```

Calculates Precision, Recall, and F1 scores for retrieved pages against ground truth.

## Design choices

**Why hybrid retrieval?**
Technical manuals have specific terms (BM25 catches these) and conceptual queries (FAISS handles semantic meaning). Combining both improves accuracy.

**Why reranking?**
Cross-encoder reranks the hybrid results for better relevance. Small performance hit but worth it.

**Why 1000 char chunks?**
Balance between context and precision. Smaller chunks = more accurate page citations but less context. 1000 works well for technical content.

**Why page-level metadata?**
Each chunk stores its source page number. When we retrieve chunks, we extract their pages. This ensures page numbers match the actual PDF.

**Why rerank_top_k=10 instead of 15?**
Tuned to reduce over-citation of pages. With fewer documents in the final context (10 vs 15), the LLM sees less irrelevant content and cites fewer unnecessary pages. The prompt also explicitly asks to only cite pages with the "main answer", not supporting info. This improves precision without hurting recall.


## Challenges faced

### 1. LLM safety blocks on certain questions
**Problem**: Some questions (like Question 1 in eval) returned empty answers because Gemini's safety filters blocked the response. Not sure why - maybe certain aviation terms triggered it.

**Solution**:
- Added retry logic with slightly rephrased prompts
- Set temperature to 0.2 for more consistent responses
- Added context in system prompt that this is for "educational/training purposes"
- Still happens occasionally - might need to switch models for those edge cases

### 2. Multimodal model selection
**Problem**: Needed image processing but had to pick the right model. Options were:
- GPT-4V: Great quality but expensive ($0.01-0.03 per image)
- Claude 3: Good but need another API key
- Gemini 1.5 Flash: Fast but less accurate
- Gemini 2.5 Pro: Best balance

**Solution**: Went with Gemini 2.5 Pro because:
- Already using Gemini ecosystem (one API key)
- Good accuracy for technical diagrams
- Reasonable cost (~$0.002 per image)
- But still made it optional (`PROCESS_IMAGES=false` by default)

### 3. API cost management with images
**Problem**: With images enabled, a 100-page manual with 200 diagrams means:
- 200+ API calls just for indexing
- Each call with images uses way more tokens
- Could easily rack up $10-20 just building the index
- Regular queries also cost more with image context

**Solution**:
- Made image processing **opt-in** (off by default)
- Cache the vector index so we only process once
- Only process images when absolutely necessary
- For most questions, text + tables are enough
- Added warning in README about costs

### 4. Question 1 returning empty results
**Problem**: First question in the eval consistently fails - no pages, no answer.

**Possible reasons**:
- Content might be in a restricted/private section
- Could be in an image-only page (no extractable text)
- Safety filters blocking it
- Specific page format not being parsed correctly

**What I tried**:
- Checked PDF - pages exist and have content
- Enabled `PROCESS_IMAGES` - helps but slow
- Tweaked prompts - no difference
- Might just be that specific question/content combination

Still investigating this one.

## Troubleshooting

**"No vector store found"**
Run once to let it build the index from the PDF.

**"API key error"**
Check your `.env` file has a valid Google API key.

**Empty pages returned**
PDF might be image-based. Try enabling `PROCESS_IMAGES=true` or use OCR preprocessing.

## Notes

- Index is built once and reused (delete `faiss_index/` to rebuild)
- Logs go to `logs/` directory
- Temperature set to 0.2 for consistent answers
- Page numbers are 1-based (matches PDF viewer)
