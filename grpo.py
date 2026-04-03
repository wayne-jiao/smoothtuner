# Warning control
import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset
from helper import generate_responses, test_model_with_questions, load_model_and_tokenizer
import re
import pandas as pd
from tqdm import tqdm

print("="*80)
print("GRPO Training Script - Step-by-Step Walkthrough")
print("="*80)
print("\n")

USE_GPU = False
print(f"📋 Configuration: USE_GPU = {USE_GPU}")
print(f"   Device: {'CUDA' if USE_GPU and torch.cuda.is_available() else 'CPU'}")
print()

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step-by-step. "
    "Always include the final numeric answer inside \\boxed{}."
)

print("="*80)
print("STEP 1: Define Reward Function")
print("="*80)

print("The reward function checks if the model's answer matches the ground truth.")
print("It extracts numbers from \\boxed{} format and gives 1.0 for correct, 0.0 for wrong.\n")

# def reward_func(completions, ground_truth, **kwargs):
#     # Regular expression to capture content inside \boxed{}
#     matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]['content']) for completion in completions]
#     contents = [match.group(1) if match else "" for match in matches]
#     # Reward 1 if the content is the same as the ground truth, 0 otherwise
#     return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

def reward_func(completions, ground_truth, **kwargs):
    """
    Improved reward function with partial credit:
    - 1.0: Correct answer in \\boxed{}
    - 0.5: Correct answer in response but not in \\boxed{}
    - 0.2: Answer appears but with extra text
    - 0.0: Wrong or missing
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        content = completion[0]['content']
        
        # First try: exact match in \boxed{}
        match = re.search(r"\\boxed\{(.*?)\}", content)
        if match:
            extracted = match.group(1).strip()
            if extracted == gt:
                rewards.append(1.0)  # Perfect!
                continue
            elif gt in extracted:
                rewards.append(0.5)  # Right answer with extra text
                continue
        
        # Second try: answer appears somewhere with word boundaries
        pattern = r'\b' + re.escape(gt) + r'\b'
        if re.search(pattern, content):
            rewards.append(0.5)  # Right answer, wrong format
            continue
        
        # Third try: answer appears anywhere
        if gt in content:
            rewards.append(0.2)  # Answer present but not isolated
            continue
        
        # No match
        rewards.append(0.0)
    
    return rewards

print("\n🧪 Testing reward function with sample predictions...")
sample_pred = [[{"role": "assistant", 
                 "content": r"...Calculating the answer. \boxed{72}"}]]
ground_truth = ["72"]
reward = reward_func(sample_pred, ground_truth)
print(f"   ✓ Correct answer (72 == 72): Reward = {reward}")

sample_pred = [[{"role": "assistant", 
                 "content": r"...Calculating the answer \boxed{71}"}]]
ground_truth = ["72"]
reward = reward_func(sample_pred, ground_truth)
print(f"   ✗ Wrong answer (71 != 72): Reward = {reward}")
print()

print("="*80)
print("STEP 2: Load Evaluation Dataset")
print("="*80)

data_num = 10  # Increased from 5 to 20 for better statistical evaluation
print(f"Loading {data_num} examples from GSM8K test set...")
eval_dataset = load_dataset("openai/gsm8k", "main")["test"].select(range(data_num))
print(f"✓ Loaded {len(eval_dataset)} examples\n")
sample_df = eval_dataset.to_pandas()
print("Sample data:")
print(sample_df.head())
print()

print("="*80)
print("STEP 3: Preprocess Data")
print("="*80)

print("Extracting ground truth answers and formatting prompts...\n")

def post_processing(example):
    match = re.search(r"####\s*(-?\d+)", example["answer"])
    example["ground_truth"] = match.group(1) if match else None
    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]}
    ]
    return example

eval_dataset = eval_dataset.map(post_processing).remove_columns(["question", "answer"])
print(f"✓ Preprocessed {len(eval_dataset)} examples")
print("\nProcessed data structure:")
sample_df = eval_dataset.select(range(min(3, len(eval_dataset)))).to_pandas()
for idx, row in sample_df.iterrows():
    print(f"  Example {idx+1}: Ground truth = {row['ground_truth']}")
print()

print("="*80)
print("STEP 4: Evaluate Base Model (Before Training)")
print("="*80)

print("Loading Qwen2.5-0.5B-Instruct model...\n")
model, tokenizer = load_model_and_tokenizer("./models/Qwen/Qwen2.5-0.5B-Instruct", USE_GPU)

print(f"\n🔍 Running baseline evaluation on {len(eval_dataset)} examples...")
print("-"*80)
# Store predictions and ground truths
all_preds = []
all_labels = []

for idx, example in enumerate(tqdm(eval_dataset, desc="Evaluating base model")):
    input_prompt = example["prompt"]
    ground_truth = example["ground_truth"]
    # Run the model to generate an answer
    with torch.no_grad():
        response = generate_responses(model, tokenizer, 
                                      full_message = input_prompt) 
    all_preds.append([{"role": "assistant", "content": response}])
    all_labels.append(ground_truth)
    print(f"\n[Example {idx+1}]")
    print(f"Response: {response}")
    print(f"Ground truth: {ground_truth}")

# 3. Evaluate using reward_func
rewards = reward_func(all_preds, all_labels)

# 4. Report accuracy
accuracy = sum(rewards) / len(rewards)
print("\n" + "="*80)
print(f"📊 BASELINE ACCURACY (before training): {accuracy:.2%}")
print(f"   Correct: {sum(rewards)}/{len(rewards)}")
print("="*80 + "\n")
del model, tokenizer

print("="*80)
print("STEP 5: Prepare Training Dataset")
print("="*80)

print("Loading full GSM8K training set...\n")
dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"]
print(f"Full training set size: {len(train_dataset)} examples")
 
# Apply to dataset
print("Preprocessing training data...")
train_dataset = train_dataset.map(post_processing)
train_dataset = train_dataset.remove_columns(["question", "answer"])
if not USE_GPU:
    train_dataset = train_dataset.select(range(50))  # Increased from 10 to 100
    print(f"⚠️  Running on CPU - using {len(train_dataset)} examples for training")
    print(f"    (This will be slower but should show better results)")
else:
    print(f"Using all {len(train_dataset)} examples for training")
print(f"\nFirst training example structure:")
print(f"  Keys: {train_dataset[0].keys()}")
print(f"  Ground truth: {train_dataset[0]['ground_truth']}")
print()

print("="*80)
print("STEP 6: Configure GRPO Training")
print("="*80)

config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=8,  # Increased from 4 to 8 for better exploration
    num_train_epochs=3,  # Increased from 1 to 3 for more training
    learning_rate=5e-6,
    logging_steps=2,
    use_cpu=not USE_GPU,  # Set to True when running on CPU
)
print("Training configuration:")
print(f"  • Batch size: {config.per_device_train_batch_size}")
print(f"  • Gradient accumulation: {config.gradient_accumulation_steps}")
print(f"  • Generations per prompt: {config.num_generations}")
print(f"  • Epochs: {config.num_train_epochs}")
print(f"  • Learning rate: {config.learning_rate}")
print(f"  • Device: {'GPU (CUDA)' if USE_GPU else 'CPU'} (use_cpu={not USE_GPU})")
print(f"  • Total training steps: ~{len(train_dataset) * config.num_train_epochs}")
print()

print("="*80)
print("STEP 7: Initialize Model for Training")
print("="*80)

print("Loading SmolLM2-135M-Instruct model for GRPO training...\n")
print("NOTE: If you have the Qwen model downloaded, it works better for math.")
print("      Change this line to: './models/Qwen/Qwen2.5-0.5B-Instruct'\n")
model, tokenizer = load_model_and_tokenizer("./models/HuggingFaceTB/SmolLM2-135M-Instruct", USE_GPU)

print("\nInitializing GRPO Trainer...")
grpo_trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=reward_func,
    train_dataset=train_dataset
)
print("✓ Trainer initialized\n")

print("="*80)
print("STEP 8: Start GRPO Training")
print("="*80)

print("Training in progress... This may take a while.\n")
grpo_trainer.train()
print("\n✓ Training completed!\n")

print("="*80)
print("STEP 9: Evaluate Trained Model")
print("="*80)

fully_trained_qwen = True
if fully_trained_qwen:
    print("Loading pre-trained GRPO model (Qwen2.5-0.5B-GRPO) for comparison...\n")
    model, tokenizer = load_model_and_tokenizer("./models/banghua/Qwen2.5-0.5B-GRPO", USE_GPU)
    
else:
    print("Using the model we just trained...\n")
    model = grpo_trainer.model

print(f"🔍 Running post-training evaluation on {len(eval_dataset)} examples...")
print("-"*80)
# Store predictions and ground truths
all_preds = []
all_labels = []

for idx, example in enumerate(tqdm(eval_dataset, desc="Evaluating trained model")):
    input_prompt = example["prompt"]
    ground_truth = example["ground_truth"]
    # Run the model to generate an answer
    with torch.no_grad():
        response = generate_responses(model, tokenizer, 
                                      full_message = input_prompt) 
    all_preds.append([{"role": "assistant", "content": response}])
    all_labels.append(ground_truth)
    print(f"\n[Example {idx+1}]")
    print(f"Response: {response}")
    print(f"Ground truth: {ground_truth}")

# 3. Evaluate using reward_func
rewards = reward_func(all_preds, all_labels)

# 4. Report accuracy
accuracy = sum(rewards) / len(rewards)
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"📊 POST-TRAINING ACCURACY: {accuracy:.2%}")
print(f"   Correct: {sum(rewards)}/{len(rewards)}")
print("="*80)
print("\n✅ Script completed successfully!")