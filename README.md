# RAG-explorer
A RAG system with vector search and retrieval pipeline. 

A RAG-style system using FAISS for vector retrieval, with an agent-like layer that routes queries and decides how to process retrieved data. Also uses validation to ensure outputs are grounded in retrieved context. Uses local free software. 


Stack:
Python 
FAISS (vector DB)
Ollama (LLM) 
LangChain
Streamlit (optional UI) 

In your README, write:


tradeoffs (quality vs cost)
limitations

Architecture:
User query
   ↓
Agent (LangChain)
   ↓ decides:
   retrieve / summarize / analyze
   ↓
Vector DB (FAISS)
   ↓
Relevant chunks
   ↓
LLM (Ollama)
   ↓
Answer
   ↓
Validation layer
   ↓
Final output


Step 0: Dataset
Pick something like:

Raw CSV
  ↓
Cleaned Documents
  ↓
Chunked Documents
  ↓
Embeddings
  ↓
FAISS Index


Step 1: Vector DB (retrieval layer)

    1. Chunk
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    2. Embed
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(chunks)


    3. FAISS storage
    import faiss
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    * Store metadata along with information for validation
    {
    "text": "...",
    "source": "...",
    "id": ...
    }

Step 2: Query and Retrieval
    embed query
    search FAISS
    return top-k chunks

Step 3: Agent Layer
    retrieve → agent decides what to do
    1. Understand user intent
        
        prompt = f"""
        User query: {query}

        Classify intent into:
        - product recommendations
        - issue detection ("what products cause breakouts?")
        - trend over time
        - compare products
        - cluster complaints

        """
        intent = llm(prompt)

    2. Decide what to retrieve
        
        if intent == "summarize":
            do_summary()
        elif intent == "analyze":
            do_analysis()

    3. Retrieve data

    4. Decide what task to perform:
    - summarize
        def summarize(chunks): ...
    - classify
    - extract insights
        def extract_issues(chunks): ...

    5. produce answer

Step 4: Validation
    1: Answer grounding
    def validate(answer, context):
        prompt = f"""
        Answer: {answer}
        Context: {context}

        Is the answer supported? Return yes/no + explanation.
        """

    2: confidence scoring
    {
    "answer": "...",
    "confidence": 0.82,
    "sources": [...]
    }

    3: schema validation
    Force structured json output, then validate with python. 


Future Work
Option A: Trend Analysis

"What are the most complained-about products this month?"

Option B: Skin concern classifier

acne / dryness / irritation clustering

Option C: Recommendation engine

"Best moisturizers for oily skin based on Reddit data"