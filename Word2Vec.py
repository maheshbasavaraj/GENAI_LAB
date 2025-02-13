# Import required libraries
import gensim.downloader as api
import numpy as np

# Load pre-trained Word2Vec model (download if not cached)
print("Loading model...")
model = api.load(
    "word2vec-google-news-300"
)  # Use "glove-wiki-gigaword-300" for a smaller model
print("Model loaded!")


# Define vector arithmetic function
def vector_arithmetic(model, positive=[], negative=[], topn=5):
    try:
        result = model.most_similar(positive=positive, negative=negative, topn=topn)
        return result
    except KeyError as e:
        return f"Word not in vocabulary: {e}"


# Example 1: King - Man + Woman = Queen
output = vector_arithmetic(model, positive=["woman", "king"], negative=["man"])
print("\nExample 1: King - Man + Woman = ?")
for word, score in output:
    print(f"{word}: {score:.4f}")

# Example 2: France - Paris + Berlin = Germany
output = vector_arithmetic(model, positive=["france", "berlin"], negative=["paris"])
print("\nExample 2: France - Paris + Berlin = ?")
for word, score in output:
    print(f"{word}: {score:.4f}")

while True:
    #exit loop if user enters 'q'
    if input("\nDo you want to try another example? (y/n): ").lower() == "n":
        break
    
    try:
        # Prompt user for input
        word1 = input("\nEnter base word (e.g., 'king'): ").strip()
        word2 = input("Enter word to subtract (e.g., 'man'): ").strip()
        word3 = input("Enter word to add (e.g., 'woman'): ").strip()
        output = vector_arithmetic(model, positive=[word1, word3], negative=[word2])
        print(f"\nResult: {word1} - {word2} + {word3} = ?")
        for word, score in output:
            print(f"{word}: {score:.4f}")
    except ValueError:
        print("Invalid input. Please enter a valid word.")
        continue


