import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_and_setup_gpt2():
    """
    Downloads and sets up GPT-2 Tiny model and tokenizer
    Returns the model and tokenizer objects
    """
    model_name = "EleutherAI/gpt-neo-125M"
    
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    """
    Generate text from a prompt
    """
    try:
        # Create input with attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True  # Explicitly request attention mask
        )
        
        # Move all inputs to the correct device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with corrected parameters
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,  # Enable sampling to use temperature
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
        
        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        return None

if __name__ == "__main__":  # Corrected syntax here
    try:
        # Download and setup
        model, tokenizer = download_and_setup_gpt2()
        
        # Test the model
        prompt = "The quick brown fox"
        print(f"\nPrompt: {prompt}")
        
        generated_text = generate_text(model, tokenizer, prompt)
        if generated_text:
            print(f"Generated: {generated_text}")
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")