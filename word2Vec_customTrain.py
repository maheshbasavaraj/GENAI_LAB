# Import libraries
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

# Step 1: Load and preprocess the domain-specific corpus
corpus_path = "movie.txt"  # Replace with your dataset

with open(corpus_path, "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize sentences and words
sentences = [
    word_tokenize(sentence.lower()) for sentence in text.split(".") if sentence
]

# Step 2: Train the Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,  # Embedding dimension
    window=,  # Context window size
    min_count=3,  # Ignore rare words
    workers=4,  # Parallel threads
    epochs=100,  # Training iterations
)

# Save the model
model.save("custom_word2vec.model")


# Step 3: Analyze embeddings
def print_similar_words(model, word):
    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=5)
        print(f"Words similar to '{word}':")
        for word, score in similar_words:
            print(f"{word}: {score:.4f}")
    else:
        print(f"'{word}' not in vocabulary.")


# Example 1: Compare domain-specific vs. general embeddings
input_word = input("Enter a word: ")
print_similar_words(model, input_word)  # Domain: biological cell

# Load a pre-trained general model (e.g., GloVe)
#glove_model = api.load("glove-wiki-gigaword-300")
#glove_model = gensim.downloader.load("glove-wiki-gigaword-300")
#print("\nGeneral model results for 'cell':")
#print(glove_model.most_similar("cell", topn=5))
