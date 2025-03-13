# Phi-2 Model Fine-tuning with LoRA

This project demonstrates how to fine-tune the Microsoft Phi-2 model using Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) and 4-bit quantization. The model is fine-tuned on the OpenAssistant Conversations Dataset (OASST1).

## Features

- 4-bit quantization using BitsAndBytes
- LoRA for parameter-efficient fine-tuning
- Integration with Hugging Face's Transformers and PEFT libraries
- Training with SFTTrainer from TRL library
- OpenAssistant dataset integration
- Model export in both Hugging Face and PyTorch formats

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Training Configuration

### Model Configuration
- Base Model: microsoft/phi-2
- Quantization: 4-bit (NF4)
- Computing dtype: float16
- Double quantization: Enabled

### LoRA Configuration
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, dense

### Training Parameters
- Number of epochs: 1
- Maximum steps: 500
- Batch size: 4
- Learning rate: 2e-4
- Weight decay: 0.001
- Gradient accumulation steps: 1
- Maximum sequence length: 512

## Usage

To start the training process, run:

```bash
python train.py
```

The fine-tuned model will be saved in the `phi2-finetuned-final` directory in both Hugging Face format and as a PyTorch state dict (model.pt).

## Model Checkpoints

During training, model checkpoints are saved:
- Every 100 steps
- Maximum of 3 saved checkpoints
- Final model saved in `phi2-finetuned-final` directory
  - Hugging Face format for compatibility with transformers library
  - PyTorch state dict saved as `model.pt` with model configuration

## Dataset

The model is fine-tuned on the OpenAssistant Conversations Dataset (OASST1), which contains high-quality conversational data.

## License

This project is open-source and follows the licensing terms of the original Phi-2 model and used libraries.