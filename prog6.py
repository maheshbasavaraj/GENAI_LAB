# Import libraries
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")  # Suppress model warnings

# Step 1: Load the sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


# Step 2: Define a function to analyze sentiment
def analyze_sentiment(text):
    if not text.strip():
        return "Error: Input text is empty."
    result = sentiment_pipeline(text)
    return result


# Step 3: Interactive input loop
print("Sentiment Analysis Tool")
print("Enter a sentence (type 'exit' to quit):")

while True:
    user_input = input("\nYour text: ").strip()
    if user_input.lower() == "exit":
        break
    if user_input:
        sentiment = analyze_sentiment(user_input)
        print("\nResult:")
        for item in sentiment:
            print(f"Label: {item['label']} | Confidence: {item['score']:.4f}")
    else:
        print("Please enter valid text.")

print("\nExiting...")
