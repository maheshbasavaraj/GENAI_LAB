# Import libraries
import gensim.downloader as api
from transformers import pipeline, set_seed
import numpy as np

# Step 1: Load pre-trained word embeddings
print("Loading word embeddings...")
embedding_model = api.load("glove-wiki-gigaword-300")

# Step 2: Function to retrieve similar words
def get_similar_words(word, topn=3):
    try:
        similar_words = embedding_model.most_similar(word, topn=topn)
        return [word for word, _ in similar_words]
    except KeyError:
        return []

# Step 3: Function to enrich the prompt
def enrich_prompt(prompt, keywords):
    enriched = f"{prompt}\nRelated keywords: {', '.join(keywords)}."
    print ("enriched prompt is ",enriched )
    return enriched

# Step 4: Load Generative AI model (GPT-2)
print("Loading GPT-2...")
generator = pipeline('text-generation', model='gpt2distilgpt2')
set_seed(42)  # For reproducibility

# Step 5: Generate responses for original and enriched prompts
def compare_prompts(base_prompt):
    # Retrieve similar words
    seed_word = "forest"  # Example seed word (customize as needed)
    similar_words = get_similar_words(seed_word)
    
    # Create enriched prompt
    enriched_prompt = enrich_prompt(base_prompt, similar_words)
    
    # Generate outputs
    original_output = generator(base_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    enriched_output = generator(enriched_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    return original_output, enriched_output

# Example usage
base_prompt = "Write a short story about a forest."
original, enriched = compare_prompts(base_prompt)

print("\n=== Original Prompt Output ===")
print("the story is ", original)

print("\n=== Enriched Prompt Output ===")
print(enriched)