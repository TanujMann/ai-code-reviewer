"""
train.py
========
Fine-tune CodeLlama-7B using LoRA/PEFT for code review.

Why LoRA?
- Full fine-tuning of 7B model needs 80GB+ VRAM
- LoRA trains only ~1% of parameters
- Same quality, runs on 16GB GPU (or Google Colab T4!)

Usage:
    python scripts/train.py
    python scripts/train.py --model codellama/CodeLlama-7b-Instruct-hf
    python scripts/train.py --model microsoft/codebert-base (smaller, CPU-friendly)
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="codellama/CodeLlama-7b-Instruct-hf",
        help="Base model to fine-tune. Use 'microsoft/phi-2' for smaller GPU."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models/code-reviewer-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (saves VRAM)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# STEP 1: LOAD MODEL WITH QUANTIZATION
# ─────────────────────────────────────────────
def load_model(model_name: str, use_4bit: bool):
    """
    Load model with optional 4-bit quantization.
    4-bit: fits CodeLlama-7B in ~8GB VRAM
    """
    logger.info(f"Loading model: {model_name}")
    
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NormalFloat4
            bnb_4bit_compute_dtype=torch.float16, # Compute in fp16
            bnb_4bit_use_double_quant=True,       # Double quantization
        )
        logger.info("✅ 4-bit quantization enabled")
    else:
        logger.info("⚠️  Running without quantization (CPU or no CUDA)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"  # Required for SFT
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    # Prepare for k-bit training (gradient checkpointing etc.)
    if use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


# ─────────────────────────────────────────────
# STEP 2: APPLY LoRA ADAPTERS
# ─────────────────────────────────────────────
def apply_lora(model):
    """
    Apply LoRA adapters to the model.
    
    LoRA injects small trainable matrices (rank r) into attention layers.
    Only these matrices get trained — everything else is frozen.
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,               # LoRA rank (higher = more params, better quality)
        lora_alpha=32,      # Scaling factor (usually 2x rank)
        target_modules=[    # Which layers to apply LoRA to
            "q_proj",       # Query projection
            "v_proj",       # Value projection
            "k_proj",       # Key projection
            "o_proj",       # Output projection
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameter stats
    trainable, total = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    
    pct = 100 * trainable / total
    logger.info(f"✅ LoRA applied!")
    logger.info(f"   Trainable params: {trainable:,} ({pct:.2f}% of total)")
    logger.info(f"   Total params:     {total:,}")
    
    return model


# ─────────────────────────────────────────────
# STEP 3: LOAD DATASET
# ─────────────────────────────────────────────
def load_training_data(data_dir: str):
    """Load the prepared JSON training data."""
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}\n"
            f"Run: python scripts/prepare_dataset.py first!"
        )
    
    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    logger.info(f"✅ Loaded {len(train_dataset)} train, {len(val_dataset)} val examples")
    return train_dataset, val_dataset


# ─────────────────────────────────────────────
# STEP 4: TRAINING ARGUMENTS
# ─────────────────────────────────────────────
def get_training_args(output_dir: str, args) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Effective batch = 2*4 = 8
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),  # Mixed precision on GPU
        bf16=False,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",             # Set to "wandb" to enable W&B logging
        dataloader_num_workers=0,     # Windows compatibility
        optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",
    )


# ─────────────────────────────────────────────
# STEP 5: TRAIN
# ─────────────────────────────────────────────
def train(args):
    # Load model + tokenizer
    model, tokenizer = load_model(args.model, args.use_4bit)
    
    # Apply LoRA
    model = apply_lora(model)
    
    # Load data
    train_dataset, val_dataset = load_training_data(args.data_dir)
    
    # Training args
    training_args = get_training_args(args.output_dir, args)
    
    # Initialize SFT Trainer (Supervised Fine-Tuning)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",      # Column containing training text
        max_seq_length=args.max_seq_length,
        packing=False,
    )
    
    logger.info("\n" + "=" * 50)
    logger.info("🚀 Starting Fine-tuning!")
    logger.info("=" * 50)
    
    # Train!
    trainer.train()
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Model saved to: {final_path}")
    logger.info(f"\nTo use your model:")
    logger.info(f"   Update backend/app/core/config.py MODEL_PATH = '{final_path}'")


# ─────────────────────────────────────────────
# STEP 6: TEST INFERENCE
# ─────────────────────────────────────────────
def test_inference(model_path: str):
    """Quick test of the fine-tuned model."""
    from peft import PeftModel
    
    logger.info("Testing inference...")
    
    base_model_name = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    
    test_code = """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""
    
    prompt = f"""### Instruction:
Review this code and identify issues.

### Code:
```python
{test_code}
```

### Review:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "=" * 50)
    print("Model Response:")
    print("=" * 50)
    print(response[len(prompt):])


if __name__ == "__main__":
    args = get_args()
    
    # First prepare data if needed
    if not os.path.exists(os.path.join(args.data_dir, "train.json")):
        logger.info("Dataset not found. Preparing dataset first...")
        from prepare_dataset import prepare_dataset
        prepare_dataset(args.data_dir)
    
    train(args)
