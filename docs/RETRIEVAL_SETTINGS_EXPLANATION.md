# Retrieval Settings - How Number of Tickets Works

## Current Configuration

**Database Status:**
- **Total Tickets**: 19 tickets (limited for privacy reasons)
- **Total Guide Sections**: 83 sections
- **Default Retrieval**: 3 tickets + 3 guide sections per query

## How Retrieval Works

### 1. **Semantic Search (Vector Search)**
The system uses **semantic similarity search** to find the most relevant tickets and guides for each query:

- **Query is converted to embedding** (vector representation using `all-mpnet-base-v2` model)
- **Vector database searches** for similar content based on meaning (not just keywords)
- **Top N results** are returned based on similarity score

### 2. **Number of Tickets Setting**

When you set `n_tickets = 3` (default):
- The system **requests the 3 most similar tickets** to the user's query
- If fewer than 3 tickets exist in total, it returns **all available tickets**
- If more tickets exist but aren't relevant, it still returns the top 3 most relevant ones

### 3. **With Limited Tickets (19 total)**

**Current Situation:**
- Database has **19 tickets total**
- Default retrieval: **3 tickets per query** (~16% of database)
- Maximum retrieval setting: **5 tickets** (~26% of database)

**How This Affects Retrieval:**

✅ **Pros:**
- With only 19 tickets, retrieving 3-5 means you're getting a good sample of potentially relevant tickets
- The semantic search ensures you get the **most relevant** tickets, not random ones
- Even with few tickets, the system prioritizes relevance over quantity

⚠️ **Considerations:**
- **Limited diversity**: With few tickets, there may be less variety in responses
- **Lower topic coverage**: Some queries might not match any tickets well
- **Higher reliance on guides**: The system compensates by relying more on technical guides (83 sections)

## Current Behavior

### When User Requests N Tickets:

1. **If N ≤ Total Tickets:**
   - Returns top N most similar tickets
   - Quality depends on semantic similarity

2. **If N > Total Tickets:**
   - Returns all available tickets (currently max 19)
   - No error - system gracefully handles this

3. **If No Relevant Tickets:**
   - Returns tickets with highest similarity (even if low)
   - LLM prompt explicitly states if context is limited
   - System suggests checking guides instead

## Recommendations for Limited Ticket Scenario

### 1. **Adjust Retrieval Settings**

Since you have limited tickets, consider:

- **Option A: Retrieve More Tickets** (Recommended for limited dataset)
  - Increase `n_tickets` to **5** (from 3)
  - This retrieves ~26% of database per query
  - Better coverage of available information
  - Trade-off: Slightly longer prompts, but better context

- **Option B: Keep Default (3 tickets)**
  - Good balance for speed
  - Still gets most relevant tickets
  - Works well when queries match tickets well

### 2. **Rely More on Guides**

With 83 guide sections, the system can:
- Retrieve 5-7 guide sections (vs 3-5 tickets)
- Use guide content as primary knowledge base
- Tickets serve as examples of past interactions, not primary knowledge

### 3. **Adjust Settings in Streamlit**

Users can adjust these in the Streamlit interface:
- **Number of tickets**: 1-5 (default: 3)
- **Number of guides**: 1-5 (default: 3)

For limited ticket scenario, consider:
- **Tickets: 5** (max available)
- **Guides: 5** (better coverage)

## Technical Details

### Similarity Threshold
- ChromaDB returns results sorted by similarity score (0-1, higher = more similar)
- No minimum threshold - always returns top N, even if similarity is low
- Average similarity for tickets: **0.47-0.52** (FAIR - this is expected with limited data)

### Fallback Behavior
- If retrieval finds low-relevance tickets, the LLM is instructed to:
  1. Use available context if relevant
  2. Clearly state if information is insufficient
  3. Suggest alternatives or checking guides
  4. Never invent information not in context

## Testing Recommendations

To verify retrieval quality with your limited dataset:

```python
# Test script to check retrieval behavior
from src.phase4.vector_db import VectorDBManager

db = VectorDBManager()
query = "Come posso lavare la mia auto?"

# Test different settings
for n_tickets in [3, 5, 10, 19]:
    results = db.search_tickets(query, n_tickets)
    print(f"Requested: {n_tickets}, Got: {len(results['ids'][0])}")
    print(f"Similarities: {results.get('distances', [])}")
```

## Summary for Client

**How it works:**
- System retrieves **top N most similar tickets** based on semantic search
- With 19 tickets, requesting 3-5 means retrieving 16-26% of database
- System **prioritizes relevance over quantity**

**Recommendations:**
- ✅ **Increase to 5 tickets** for better coverage (if speed allows)
- ✅ **Increase guides to 5** to compensate for limited tickets
- ✅ **Monitor response quality** - system handles limited data gracefully
- ⚠️ **Add more tickets** over time to improve coverage and variety

**Current System Handles:**
- ✅ Automatically returns all available tickets if less than requested
- ✅ Uses guides as primary knowledge base (83 sections available)
- ✅ LLM explicitly states when information is insufficient
- ✅ Never invents information - only uses retrieved context

## Questions?

If you have concerns about:
- **Privacy**: Tickets are stored locally in ChromaDB, not sent to external APIs
- **Quality**: System works well even with limited tickets, relies more on guides
- **Coverage**: Consider increasing retrieval to 5 tickets + 5 guides for better context


