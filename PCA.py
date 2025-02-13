# Import libraries
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load pre-trained embeddings
model = api.load("glove-wiki-gigaword-300")  # Replace with your model

# Step 1: Select 10 domain-specific words (e.g., technology)
domain_words = [
    "computer",
    "software",
    "algorithm",
    "data",
    "network",
    "robot",
    "artificial",
    "internet",
    "cloud",
    "encryption",
]

# Step 2: Extract embeddings for these words
embeddings = np.array([model[word] for word in domain_words if word in model])

# Step 3: Dimensionality reduction using PCA or t-SNE
# Option 1: PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Option 2: t-SNE (uncomment below)
# tsne = TSNE(n_components=2, random_state=42)
# reduced_embeddings = tsne.fit_transform(embeddings)

# Step 4: Plot the embeddings
plt.figure(figsize=(10, 8))
for i, word in enumerate(domain_words):
    if word in model:
        x, y = reduced_embeddings[i]
        plt.scatter(x, y)
        plt.text(x, y, word, fontsize=9)
plt.title("2D Visualization of Word Embeddings (PCA)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


# Step 5: Generate 5 semantically similar words for a given input
def get_similar_words(model, word, topn=5):
    try:
        return model.most_similar(word, topn=topn)
    except KeyError:
        return f"'{word}' not in vocabulary."


# Example usage
#take input word
# get 5 most similar words

input_word = input("Enter a word: ")
similar_words = get_similar_words(model, input_word)
print(f"\nTop 5 words similar to '{input_word}':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

