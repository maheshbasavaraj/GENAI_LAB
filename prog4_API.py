# Import libraries
import gensim.downloader as api
import openai
import os

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key in environment variables
# OR directly:
# openai.api_key = "sk-your-api-key-here"

# Step 1: Load pre-trained word embeddings (same as before)
print("Loading word embeddings...")
embedding_model = api.load("glove-wiki-gigaword-300")

# Step 2: Function to retrieve similar words (same as before)
def get_similar_words(word, topn=3):
    try:
        similar_words = embedding_model.most_similar(word, topn=topn)
        return [word for word, _ in similar_words]
    except KeyError:
        return []

# Step 3: Function to enrich the prompt (same as before)
def enrich_prompt(prompt, keywords):
    enriched = f"{prompt}\nRelated keywords: {', '.join(keywords)}."
    return enriched

# Step 4: OpenAI text generation function
def generate_text(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content

# Step 5: Modified compare_prompts function
def compare_prompts(base_prompt):
    seed_word = "forest"
    similar_words = get_similar_words(seed_word)
    
    enriched_prompt = enrich_prompt(base_prompt, similar_words)
    
    original_output = generate_text(base_prompt)
    enriched_output = generate_text(enriched_prompt)
    
    return original_output, enriched_output

# Example usage (same as before)
base_prompt = "Write a short story about a forest."
original, enriched = compare_prompts(base_prompt)

print("\n=== Original Prompt Output ===")
print(original)

print("\n=== Enriched Prompt Output ===")
print(enriched)