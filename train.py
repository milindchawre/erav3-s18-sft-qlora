from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model and tokenizer
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "dense"]
)

# Load and preprocess dataset
ds = load_dataset("OpenAssistant/oasst1")
train_dataset = ds['train']

def format_conversation(example):
    """Format the conversation for instruction fine-tuning"""
    # Only process root messages (start of conversations)
    if example["role"] == "prompter" and example["parent_id"] is None:
        conversation = []
        current_msg = example
        conversation.append(("Human", current_msg["text"]))
        
        # Follow the conversation thread
        current_id = current_msg["message_id"]
        while current_id in message_children:
            # Get the next message in conversation
            next_msg = message_children[current_id]
            if next_msg["role"] == "assistant":
                conversation.append(("Assistant", next_msg["text"]))
            elif next_msg["role"] == "prompter":
                conversation.append(("Human", next_msg["text"]))
            current_id = next_msg["message_id"]
            
        if len(conversation) >= 2:  # At least one exchange (human->assistant)
            formatted_text = ""
            for speaker, text in conversation:
                formatted_text += f"{speaker}: {text}\n\n"
            return {"text": formatted_text.strip()}
    return {"text": None}

# Build message relationships
print("Building conversation threads...")
message_children = {}
for example in train_dataset:
    if example["parent_id"] is not None:
        message_children[example["parent_id"]] = example

# Format complete conversations
print("\nFormatting conversations...")
processed_dataset = []
for example in train_dataset:
    result = format_conversation(example)
    if result["text"] is not None:
        processed_dataset.append(result)
    if len(processed_dataset) % 100 == 0 and len(processed_dataset) > 0:
        print(f"Found {len(processed_dataset)} valid conversations")

print(f"Final dataset size: {len(processed_dataset)} conversations")

# Convert to Dataset format
train_dataset = Dataset.from_list(processed_dataset)

# Remove the redundant conversion
# train_dataset = list(train_dataset)
# train_dataset = Dataset.from_list(train_dataset)

# Convert to standard dataset for training
train_dataset = list(train_dataset)
train_dataset = Dataset.from_list(train_dataset)

# Configure SFT parameters
sft_config = SFTConfig(
    output_dir="phi2-finetuned",
    num_train_epochs=1,
    max_steps=500,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    logging_steps=1,
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    push_to_hub=False,
    max_seq_length=512,
    report_to="none",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,  # Changed from dataset to train_dataset
    peft_config=peft_config,
    args=sft_config,
)

# Train the model
trainer.train()

# Save the trained model in Hugging Face format
trainer.save_model("phi2-finetuned-final")

# Save the model in PyTorch format
model_save_path = "phi2-finetuned-final/model.pt"
torch.save({
    'model_state_dict': trainer.model.state_dict(),
    'config': trainer.model.config,
    'peft_config': peft_config,
}, model_save_path)
print(f"Model saved in PyTorch format at: {model_save_path}")
