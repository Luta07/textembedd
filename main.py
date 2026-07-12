import os
import math
from fastapi import FastAPI, Request
from openai import AsyncOpenAI

app = FastAPI()

# Initialize OpenAI client configured for AI Pipe
client = AsyncOpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude_vec1 = math.sqrt(sum(a * a for a in vec1))
    magnitude_vec2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    return dot_product / (magnitude_vec1 * magnitude_vec2)

@app.post("/")
async def rank_candidates(request: Request):
    data = await request.json()
    query = data.get("query", "")
    candidates = data.get("candidates", [])
    
    if not query or not candidates:
        return {"error": "Missing query or candidates"}

    # Batch the query and all candidates into a single request
    texts_to_embed = [query] + candidates
    
    try:
        # Use AI Pipe to route the embedding request with the provider prefix
        response = await client.embeddings.create(
            input=texts_to_embed,
            model="openai/text-embedding-3-small" 
        )
        
        # Extract the embeddings from the response 
        embeddings = [item.embedding for item in response.data]
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Calculate cosine similarity for each candidate
        scores = []
        for i, emb in enumerate(candidate_embeddings):
            score = cosine_similarity(query_embedding, emb)
            scores.append((score, i))
            
        # Sort candidates by similarity score in descending order
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Extract the original indices of the top 3 candidates
        top_3_indices = [idx for score, idx in scores[:3]]
        
        # Return exactly the JSON structure the grader expects
        return {"ranking": top_3_indices}
        
    except Exception as e:
        # Catch and return any errors so the grader/Render logs show what went wrong
        return {"error": str(e)}
