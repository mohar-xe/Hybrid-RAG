from dotenv import load_dotenv
from config.constants import BASE_URL, GENERATOR_MODEL
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("NIM_API_KEY")) 

def generate_answer(user_query, retrieved_chunks):
    print("Generating answer...\n")
    context_text = "\n\n---\n\n".join(
        [chunk[0] for chunk in retrieved_chunks]
    )
    
    prompt = f"""
    You are a helpful and knowledgeable AI assistant. 
    Use the following retrieved context to answer the user's question. 
    If the answer is not contained in the context, say "I cannot answer this based on the provided documents."

    Context:
    {context_text}

    Question:
    {user_query}
    """

    client = OpenAI(
    base_url = BASE_URL,
    api_key = os.getenv("NIM_API_KEY")
    )

    response = client.chat.completions.create(
    model=GENERATOR_MODEL,
    messages=[{"role":"system","content":"/think"},
              {"role": "user", "content": prompt}],
    temperature=0.2,
    top_p=0.95,
    max_tokens=65536,
    frequency_penalty=0,
    presence_penalty=0,
    stream=False
    )
    
    return response.choices[0].message.content