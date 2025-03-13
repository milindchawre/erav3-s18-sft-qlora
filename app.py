import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and tokenizer
model_name = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},  # Force CPU usage
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(
    base_model,
    "phi2-finetuned-final",
    device_map={"": "cpu"}  # Force CPU usage
)

def generate_response(message, history):
    # Format input as instruction-based conversation
    prompt = "You are a helpful AI assistant. Please provide clear and concise responses.\n\n"
    for human, assistant in history[-7:]:  # Keep last 7 exchanges for context
        prompt += f"Instruction: {human}\nResponse: {assistant}\n\n"
    prompt += f"Instruction: {message}\nResponse:"

    # Generate response with limited length
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,     # Limited to 64 tokens
            max_length=512,        # Keep history context at 512
            temperature=0.6,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.7,
            min_length=1,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Create Gradio interface
css = """
.gradio-container {max-width: 1000px !important}
.chatbot {min-height: 700px !important}
.chat-message {font-size: 16px !important}
"""

demo = gr.ChatInterface(
    generate_response,
    chatbot=gr.Chatbot(height=700),  # Increased height
    textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=0.9),
    title="Phi-2 Conversational Assistant",
    description="A fine-tuned Phi-2 model for conversational AI",
    theme="soft",
    css=css,
    examples=["Tell me about yourself",
             "What can you help me with?",
             "How do you process information?"],
)

if __name__ == "__main__":
    demo.launch(share=True)