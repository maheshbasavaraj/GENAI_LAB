# Import libraries
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")  # Suppress model warnings

# Step 1: Load the summarization pipeline
summarizer = pipeline(
    "summarization", model="facebook/bart-large-cnn"  # Optimized for summarization
)


# Step 2: Define a function to summarize text
def summarize_text(text):
    if not text.strip():
        return "Error: Input text is empty."
    if len(text.split()) < 30:
        return "Error: Input text is too short (min 30 words)."

    # Generate summary with configurable length
    summary = summarizer(
        text,
        max_length=150,  # Maximum summary length
        min_length=30,  # Minimum summary length
        do_sample=False,  # Disable random sampling for deterministic output
    )
    return summary[0]["summary_text"]


# Step 3: Interactive input loop
print("Text Summarization Tool")
print("Enter a passage (type 'exit' to quit):")

while True:
    user_input = input("\nYour text: ").strip()
    if user_input.lower() == "exit":
        break
    if user_input:
        summary = summarize_text(user_input)
        print("\n=== Summary ===")
        print(summary)
    else:
        print("Please enter valid text.")

print("\nExiting...")
