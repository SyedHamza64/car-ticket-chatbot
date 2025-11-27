# Alternative: Converting to Flask for Shared Hosting

If you **must** use shared hosting, you'll need to:

1. **Convert Streamlit to Flask/FastAPI** (simpler web framework)
2. **Use external LLM API** instead of local Ollama (OpenAI, Anthropic, etc.)
3. **Host vector DB externally** (Pinecone, Weaviate Cloud, etc.)

## Why This Approach?

- Shared hosting can run Flask/FastAPI via CGI/FastCGI
- External APIs don't require local services
- Cloud vector DBs are managed services

## Conversion Steps

### Step 1: Install Flask

```bash
pip install flask flask-cors
```

### Step 2: Create Flask App

Create `app.py` (replaces `streamlit_app.py`):

```python
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from src.phase4.rag_pipeline import RAGPipeline

app = Flask(__name__)
CORS(app)

# Initialize pipeline (use external LLM API)
pipeline = RAGPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    language = data.get('language', 'italian')
    n_tickets = data.get('n_tickets', 3)
    n_guides = data.get('n_guides', 2)
    
    result = pipeline.query(
        query=query_text,
        n_tickets=n_tickets,
        n_guides=n_guides,
        language=language
    )
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False)
```

### Step 3: Modify RAG Pipeline for External LLM

Update `src/phase4/rag_pipeline.py` to use OpenAI/Anthropic instead of Ollama:

```python
# Replace ollama with openai
import openai

# In query method:
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or gpt-4
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=temperature
)
```

### Step 4: Use Cloud Vector DB

Replace ChromaDB with Pinecone or Weaviate:

```python
# Install: pip install pinecone-client
import pinecone

pinecone.init(api_key="your-key", environment="us-east1")
index = pinecone.Index("tickets")
```

### Step 5: Deploy to Shared Hosting

1. Upload files via FTP/SFTP
2. Create `.htaccess` for Flask (if Apache):
```apache
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteRule ^(.*)$ app.py/$1 [QSA,L]
```

3. Configure Python app in cPanel
4. Set environment variables in cPanel

## Cost Estimate

- Shared Hosting: $2-7/month
- OpenAI API: ~$0.002 per query (varies)
- Pinecone: Free tier or $70/month
- **Total**: ~$10-80/month depending on usage

## Limitations

- ❌ Higher latency (external APIs)
- ❌ Ongoing API costs
- ❌ Less control over LLM
- ❌ More complex setup
- ✅ Can use shared hosting

## Recommendation

**Still recommend VPS** - better performance, lower long-term costs, more control.





