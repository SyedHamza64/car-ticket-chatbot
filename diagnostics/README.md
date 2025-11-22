# RAG System Diagnostics Suite

Comprehensive testing and profiling tools to measure and optimize the RAG pipeline.

## Overview

This suite provides three types of tests:

1. **Latency Profiler** - Measures timing breakdown of each pipeline step
2. **Retrieval Quality Test** - Evaluates vector search accuracy
3. **Response Quality Test** - Assesses LLM output quality, hallucinations, and completeness

## Quick Start

Run all diagnostics in one go:

```bash
cd E:\work\chat-bot-ticket
python diagnostics/run_full_diagnostic.py
```

This will:
- Profile latency across multiple queries
- Test retrieval quality (embedding accuracy)
- Test response quality (hallucination detection)
- Generate prioritized optimization recommendations

## Results Location

All results are saved to `diagnostics/results/`:
- `full_diagnostic.json` - Complete test results
- `latency_profile.json` - Detailed timing breakdown
- `retrieval_quality.json` - Vector search metrics
- `response_quality.json` - LLM response analysis
- `RECOMMENDATIONS.txt` - Prioritized action items

## Individual Tests

### 1. Latency Profiler

Measures time spent in each step of the RAG pipeline:

```bash
python diagnostics/latency_profiler.py
```

**Metrics:**
- Query embedding time
- Vector search time (tickets + guides)
- Context formatting time
- Prompt creation time
- LLM generation time
- Token counts (estimated)

### 2. Retrieval Quality Test

Tests how well vector search retrieves relevant context:

```bash
python diagnostics/retrieval_quality_test.py
```

**Metrics:**
- Similarity scores (tickets vs guides)
- Topic coverage (% docs containing expected keywords)
- Best match quality
- Retrieval precision

### 3. Response Quality Test

Evaluates LLM response quality and accuracy:

```bash
python diagnostics/response_quality_test.py
```

**Metrics:**
- Required keyword coverage
- Hallucination detection (forbidden keywords)
- Response completeness
- Professional tone (greeting/closing)
- Structure (paragraph count)
- Product mention verification
- Overall quality score (0-100)

## Understanding Results

### Latency Breakdown

**Good:**
- LLM time: 70-85% of total
- Search time: <10%
- Embedding time: <5%

**Bad:**
- LLM time: >90% (model too slow)
- Search time: >15% (indexing issues)
- Total time: >15s (unacceptable for production)

### Retrieval Quality

**Good:**
- Average similarity: >0.5
- Topic coverage: >60%

**Bad:**
- Average similarity: <0.3
- Topic coverage: <40%

### Response Quality

**Good:**
- Overall score: >70/100
- Keyword coverage: >70%
- Hallucination: <10%
- Completeness: 100%

**Bad:**
- Overall score: <50/100
- Keyword coverage: <50%
- Hallucination: >30%

## Expected Runtime

- Latency Profiler: ~2-3 minutes (5 queries × 2 runs)
- Retrieval Quality: ~30 seconds (7 queries, no LLM)
- Response Quality: ~3-5 minutes (5 queries with LLM)
- **Full Diagnostic: ~6-8 minutes total**

## Optimization Workflow

1. Run full diagnostic: `python diagnostics/run_full_diagnostic.py`
2. Review `RECOMMENDATIONS.txt` for prioritized actions
3. Apply immediate fixes (parameter changes)
4. Re-run diagnostics to measure improvement
5. Implement short-term optimizations
6. Repeat testing

## Example Output

```
================================================================================
OPTIMIZATION RECOMMENDATIONS (Prioritized)
================================================================================

🔥 IMMEDIATE ACTIONS (Do Now - Hours)
================================================================================

1. [CRITICAL] Latency: Stop generating 3 drafts in parallel - generate 1 draft only
   Issue: Very high latency: 13.2s average
   Impact: 40-60% reduction in LLM time (8s → ~3.5-5s)
   How: Set num_drafts=1 in Streamlit app
   Reference: Quick Win #1

2. [HIGH] Latency: Reduce num_predict from 500 to 200-250 and num_ctx to 1024
   Issue: LLM takes 88% of total time
   Impact: 20-40% faster generation
   How: Update generate_response() in rag_pipeline.py
   Reference: Quick Win #2

================================================================================
RECOMMENDED PARAMETER CHANGES (Apply Immediately)
================================================================================

In src/phase4/rag_pipeline.py → generate_response():
  num_predict: 500 → 200-250
  num_ctx: 1536 → 1024
  temperature: 0.7 → 0.3-0.5
  top_k: 50 → 40

In streamlit_app.py:
  num_drafts: 3 → 1 (default)
```

## Troubleshooting

**"ModuleNotFoundError"**
- Ensure you're in the project root
- Run: `pip install -r requirements.txt`

**"ChromaDB not found"**
- Populate database first: `python scripts/run_phase4_setup.py`

**Tests take too long**
- Reduce query count in individual test files
- Run individual tests instead of full diagnostic

**LLM errors**
- Ensure Ollama is running: `ollama list`
- Check model is available: `ollama pull gemma2:2b`

## Next Steps

After diagnostics, refer to the optimization guide provided by the user for implementation details on:

- Quick wins (hours)
- Near-term improvements (1-2 weeks)
- Long-term enhancements (1+ months)

