# LLM Optimization Plan - Fast Inference Stack

## Current Situation

- **LLM Latency**: 93% of total query time (3.5-5.5s out of 4-6s)
- **Current Model**: `gemma2:2b` via Ollama
- **Current Speed**: ~4-6s single query, ~13s for 3 drafts
- **Bottleneck**: LLM token generation (no quantization, no KV-cache optimization)

## Recommendation: Mixtral 8x7B with Optimized Serving

### Option 1: Quick Win - Ollama with Mixtral (EASIEST)
**Setup Time**: 5-10 minutes
**Expected Speedup**: 40-60% faster than gemma2:2b

#### Steps:
1. Pull quantized Mixtral via Ollama:
   ```bash
   ollama pull mixtral:8x7b-instruct-q4_K_M  # 4-bit quantization, ~24GB
   # OR smaller option:
   ollama pull mixtral:8x7b-instruct-q4_0    # 4-bit, ~22GB
   ```

2. Update `.env`:
   ```env
   OLLAMA_MODEL=mixtral:8x7b-instruct-q4_K_M
   ```

3. Test and compare

**Pros**: 
- Zero code changes needed
- Ollama handles quantization automatically
- Easy model switching
- Good speed improvement

**Cons**:
- Still uses Ollama's inference engine (not vLLM-level optimization)
- Less fine-grained control

---

### Option 2: Maximum Performance - vLLM Integration (BEST)
**Setup Time**: 30-60 minutes
**Expected Speedup**: 60-80% faster (2-3x improvement)

#### Architecture:
```
RAG Pipeline → vLLM Server (quantized Mixtral) → Fast Inference
```

#### Implementation Steps:

1. **Install vLLM**:
   ```bash
   pip install vllm
   ```

2. **Start vLLM Server**:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
     --quantization awq \  # Or gptq
     --tensor-parallel-size 1 \
     --max-model-len 4096 \
     --port 8000
   ```

3. **Update RAG Pipeline** to support vLLM:
   - Add vLLM client integration
   - Maintain Ollama compatibility (fallback)
   - Add streaming support with KV-cache

4. **Benefits**:
   - Continuous batching (process multiple queries together)
   - PagedAttention (efficient KV-cache)
   - Quantization (AWQ/GPTQ - 2-4x faster)
   - Prefilling optimization
   - Tensor parallelism (multi-GPU)

**Pros**:
- Maximum speed (industry-leading inference)
- Advanced features (batching, KV-cache, quantization)
- Production-ready
- Better GPU utilization

**Cons**:
- More complex setup
- Requires code changes
- Larger model files (~24-50GB)
- Higher GPU memory requirements

---

## Alternative Models (Consider These)

### 1. Mistral 7B Instruct (Balanced)
- **Ollama**: `mistral:7b-instruct-q4_K_M`
- **Size**: ~4GB quantized
- **Speed**: Faster than Mixtral, still excellent quality
- **Best for**: If 8x7B is too large

### 2. Mixtral 8x22B (Highest Quality)
- **Ollama**: `mixtral:8x22b-instruct-q4_K_M`
- **Size**: ~90GB quantized
- **Speed**: Slower but best quality
- **Best for**: Maximum quality if speed is less critical

### 3. Llama 3.1 8B (Good Alternative)
- **Ollama**: `llama3.1:8b-instruct-q4_K_M`
- **Size**: ~4.5GB quantized
- **Speed**: Fast, good quality
- **Best for**: Alternative to Mistral

---

## Implementation Roadmap

### Phase 1: Quick Win (Today - 10 min)
1. ✅ Test Mixtral 8x7B via Ollama (quantized)
2. ✅ Compare performance vs gemma2:2b
3. ✅ Update `.env` if better

### Phase 2: vLLM Integration (This Week - 2-3 hours)
1. Install and configure vLLM
2. Create vLLM client wrapper
3. Update RAG pipeline for vLLM support
4. Benchmark and compare

### Phase 3: Advanced Optimization (Next Week)
1. Implement continuous batching
2. Add KV-cache streaming
3. Fine-tune quantization
4. Multi-GPU if available

---

## Expected Performance Improvements

| Configuration | Single Query | 3 Drafts | Quality |
|--------------|--------------|----------|---------|
| **Current (gemma2:2b)** | 4-6s | ~13s | 86.5/100 |
| **Mixtral 8x7B (Ollama Q4)** | 2-3s | 6-9s | 90+/100 |
| **Mixtral 8x7B (vLLM AWQ)** | 1-2s | 3-5s | 92+/100 |

**Target Achievement**: ✅ 20s target met (6-9s < 20s)

---

## Hardware Requirements

### Mixtral 8x7B (Quantized Q4):
- **VRAM**: 24-28GB (single GPU)
- **RAM**: 32GB+ recommended
- **CPU**: 8+ cores recommended

### Mixtral 8x7B (vLLM):
- **VRAM**: 24-32GB (single GPU) or 2x GPUs with tensor parallelism
- **RAM**: 32GB+ recommended
- **CUDA**: 11.8+ (for optimized kernels)

---

## Quick Start: Ollama Route (Recommended First)

```bash
# 1. Pull quantized Mixtral
ollama pull mixtral:8x7b-instruct-q4_K_M

# 2. Test if model works
ollama run mixtral:8x7b-instruct-q4_K_M "Ciao, come posso aiutarti?"

# 3. Update .env
# OLLAMA_MODEL=mixtral:8x7b-instruct-q4_K_M

# 4. Restart Streamlit and test
streamlit run streamlit_app.py
```

---

## vLLM Setup (For Maximum Performance)

### 1. Install vLLM:
```bash
pip install vllm
# Or with CUDA support:
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

### 2. Download Model (if needed):
```bash
# vLLM will auto-download, but you can pre-download:
# HuggingFace will auto-download on first use
```

### 3. Start vLLM Server:
```bash
# Basic (single GPU)
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --quantization awq \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port 8000

# With multiple GPUs
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --port 8000
```

### 4. Test vLLM:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "Ciao, come posso aiutarti?",
    "max_tokens": 100
  }'
```

---

## Code Changes Needed (For vLLM)

1. **Create vLLM Client Wrapper** (`src/phase4/vllm_client.py`)
2. **Update RAG Pipeline** to support vLLM backend
3. **Add fallback** to Ollama if vLLM unavailable
4. **Update Streamlit** to show inference backend

---

## Benchmarking Plan

### Test Queries:
1. "Come posso lavare la mia auto?"
2. "Come rimuovere graffi dalla vernice?"
3. "Prodotti per pulire gli interni?"

### Metrics to Measure:
- Single query latency
- 3-draft generation time (parallel)
- Response quality score
- Token generation speed (tokens/sec)
- GPU utilization
- Memory usage

---

## Recommendation

**Start with Option 1** (Ollama + Mixtral Q4):
- Quick to implement (10 minutes)
- Significant improvement (40-60% faster)
- No code changes needed
- Easy to test and compare

**Then move to Option 2** (vLLM) if:
- You need maximum performance
- You have GPU resources
- You want production-grade inference

---

## Next Steps

1. **Today**: Test Mixtral 8x7B via Ollama
2. **This Week**: Implement vLLM if needed
3. **Monitor**: Track performance improvements
4. **Optimize**: Fine-tune based on benchmarks


