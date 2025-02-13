import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def download_and_setup_gpt2():
    """
    Downloads and sets up GPT-2 Tiny model and tokenizer
    Returns the model and tokenizer objects
    """
    # Download and load the tokenizer
    print("Downloading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    
    # Download and load the model
    print("Downloading model...")
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    """
    Generate text from a prompt
    """
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
if __name__ == "__main__":
    # First make sure you have the required packages:
    # pip install torch transformers
    
    # Download and setup
    model, tokenizer = download_and_setup_gpt2()
    
    # Try generating some text
    prompt = "The quick brown fox"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
