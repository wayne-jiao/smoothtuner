# GRPO Training for Futures Contract Specification Extraction
# 
# This script trains a language model to extract contract specifications
# from futures contract documents using Group Relative Policy Optimization (GRPO).
#
# Prerequisites:
# 1. Run generate_grpo_dataset.py to create the training dataset
# 2. Ensure the following datasets exist:
#    - ./futures_grpo_train (training set)
#    - ./futures_grpo_test (evaluation set)
#
# The model learns to:
# - Read contract specification documents
# - Extract specific fields (lot_size, currency, exchange, etc.)
# - Format answers in \\boxed{} notation
#
# TROUBLESHOOTING MODEL COLLAPSE:
# If the model outputs repetitive nonsense (e.g., "API/API/API..."):
# 1. Reduce learning_rate in GRPOConfig (try 5e-7 or 1e-7)
# 2. Reduce gradient_accumulation_steps (try 2)
# 3. Reduce num_generations (try 2)
# 4. Use only 1 epoch
# 5. Use fewer training examples (10-20) to test
# 6. Reduce temperature in generation_config (try 0.5)
# 7. Reduce max_new_tokens in generation_config (try 128)
# 8. Check if base model is working before training
#
# Warning control
import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset, load_from_disk
from helper import generate_responses, test_model_with_questions, load_model_and_tokenizer
import re
import pandas as pd
from tqdm import tqdm

print("="*80)
print("GRPO Training Script - Futures Contract Specification Extraction")
print("="*80)
print("\n")

USE_GPU = False
print(f"📋 Configuration: USE_GPU = {USE_GPU}")
print(f"   Device: {'CUDA' if USE_GPU and torch.cuda.is_available() else 'CPU'}")
print()

SYSTEM_PROMPT = (
    "You are an expert at reading and analyzing futures contract specifications. "
    "You will be provided with a contract specification document and asked specific questions about it. "
    "Analyze the document carefully and provide precise answers. "
    "Always include your final answer inside \\boxed{}."
)

print("="*80)
print("STEP 1: Define Reward Function")
print("="*80)

print("The reward function checks if the model's extracted answer matches the ground truth.")
print("It extracts content from \\boxed{} format and gives partial credit for correct answers.\n")

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
    
    Also includes collapse detection.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        content = completion[0]['content']
        
        # Detect model collapse: repetitive tokens
        words = content.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words = likely collapse
                rewards.append(0.0)
                continue
        
        # Detect excessive special characters (another collapse signal)
        if content.count('/') > 20 or content.count('API') > 5:
            rewards.append(0.0)
            continue
        
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
                 "content": r"Looking at the specification, the exchange is \\boxed{XSFE}"}]]
ground_truth = ["XSFE"]
reward = reward_func(sample_pred, ground_truth)
print(f"   ✓ Correct answer (XSFE == XSFE): Reward = {reward}")

sample_pred = [[{"role": "assistant", 
                 "content": r"The exchange code is \\boxed{XCME}"}]]
ground_truth = ["XSFE"]
reward = reward_func(sample_pred, ground_truth)
print(f"   ✗ Wrong answer (XCME != XSFE): Reward = {reward}")
print()

print("="*80)
print("STEP 2: Load Evaluation Dataset")
print("="*80)

data_num = 20  # Number of evaluation examples
print(f"Loading {data_num} examples from futures test set...")
try:
    eval_dataset = load_from_disk("./futures_grpo_test")
    if len(eval_dataset) > data_num:
        eval_dataset = eval_dataset.select(range(data_num))
    print(f"✓ Loaded {len(eval_dataset)} examples\n")
except FileNotFoundError:
    print("❌ Error: futures_grpo_test dataset not found!")
    print("   Please run generate_grpo_dataset.py first to create the dataset.\n")
    exit(1)

sample_df = eval_dataset.to_pandas()
print("Sample data:")
print(sample_df.head())
print()

print("="*80)
print("STEP 3: Check Data Format")
print("="*80)

print("The futures dataset is already preprocessed with 'prompt' and 'ground_truth' fields.\n")
print(f"✓ Dataset has {len(eval_dataset)} examples")
print("\nData structure:")
sample_df = eval_dataset.select(range(min(3, len(eval_dataset)))).to_pandas()
for idx, row in sample_df.iterrows():
    print(f"  Example {idx+1}: Ground truth = {row['ground_truth']}")
    print(f"              RIC Root = {row['metadata']['ric_root']}")
    print(f"              Field = {row['metadata']['field']}")
print()

print("="*80)
print("STEP 4: Evaluate Base Model (Before Training)")
print("="*80)

print("Loading Qwen2.5-0.5B-Instruct base model for baseline evaluation...\n")
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

print("Loading futures GRPO training set...\n")
try:
    train_dataset = load_from_disk("./futures_grpo_train")
    print(f"Full training set size: {len(train_dataset)} examples")
except FileNotFoundError:
    print("❌ Error: futures_grpo_train dataset not found!")
    print("   Please run generate_grpo_dataset.py first to create the dataset.\n")
    exit(1)

# The futures dataset is already preprocessed
if not USE_GPU:
    # Use fewer examples for CPU training
    original_size = len(train_dataset)
    train_dataset = train_dataset.select(range(min(20, len(train_dataset))))  # Reduced from 50
    print(f"⚠️  Running on CPU - using {len(train_dataset)}/{original_size} examples for training")
    print(f"    (Using fewer examples to prevent overfitting and collapse)")
else:
    # Even on GPU, consider using a subset first to test
    print(f"Using all {len(train_dataset)} examples for training")
    print(f"   Note: Consider using train_dataset.select(range(100)) for initial testing")

print(f"\nFirst training example structure:")
print(f"  Keys: {train_dataset[0].keys()}")
print(f"  Ground truth: {train_dataset[0]['ground_truth']}")
print(f"  RIC Root: {train_dataset[0]['metadata']['ric_root']}")
print(f"  Field: {train_dataset[0]['metadata']['field']}")
print()

print("="*80)
print("STEP 6: Configure GRPO Training")
print("="*80)

config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Reduced from 8 to prevent overfitting
    num_generations=4,  # Reduced from 8 - less exploration, more stable
    num_train_epochs=1,  # Reduced from 3 - one epoch is often enough for RL
    learning_rate=1e-6,  # Reduced from 5e-6 to prevent collapse
    logging_steps=2,
    use_cpu=not USE_GPU,
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

print("⚠️  Conservative settings to prevent model collapse")
print("   Generation parameters (max_length, temperature) set in Step 7")
print()

print("="*80)
print("STEP 7: Initialize Model for Training")
print("="*80)

print("Loading Qwen2.5-0.5B-Instruct model for GRPO training...\n")
print("NOTE: This model works well for extracting information from documents.")
print("      You can also try SmolLM2-135M-Instruct for a smaller/faster model.\n")
model, tokenizer = load_model_and_tokenizer("./models/Qwen/Qwen2.5-0.5B-Instruct", USE_GPU)

# Configure generation to prevent collapse
print("Configuring generation parameters to prevent model collapse...")
model.generation_config.max_length = 512  # Limit response length
model.generation_config.max_new_tokens = 256  # Limit new tokens
model.generation_config.temperature = 0.7  # More focused outputs (lower = less random)
model.generation_config.do_sample = True
model.generation_config.top_p = 0.9
print(f"  • Max length: {model.generation_config.max_length}")
print(f"  • Max new tokens: {model.generation_config.max_new_tokens}")
print(f"  • Temperature: {model.generation_config.temperature}")
print()

print("Initializing GRPO Trainer...")
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
print("⚠️  Monitoring for model collapse (repetitive/nonsensical outputs)")
print("   If you see degradation, stop training (Ctrl+C) and adjust hyperparameters\n")

# Quick sanity check before training
print("🔍 Pre-training sanity check...")
test_prompt = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is the exchange code for CME? Answer briefly."}
]
with torch.no_grad():
    test_output = generate_responses(model, tokenizer, full_message=test_prompt)
print(f"   Test output: {test_output[:100]}...")
if len(set(test_output.split())) < 5 or test_output.count('/') > 10:
    print("   ⚠️  Warning: Model might already be degraded!")
else:
    print("   ✓ Model looks healthy before training\n")

grpo_trainer.train()
print("\n✓ Training completed!\n")

# Post-training sanity check
print("🔍 Post-training sanity check...")
with torch.no_grad():
    test_output = generate_responses(grpo_trainer.model, tokenizer, full_message=test_prompt)
print(f"   Test output: {test_output[:100]}...")
if len(set(test_output.split())) < 5 or test_output.count('/') > 10:
    print("   ❌ WARNING: Model has collapsed! Outputs are repetitive.")
    print("   → Reduce learning_rate, num_generations, or num_train_epochs")
    print("   → Increase kl_coef (e.g., 0.2 or 0.5)")
else:
    print("   ✓ Model still generating diverse outputs\n")

print("="*80)
print("STEP 9: Evaluate Trained Model")
print("="*80)

print("Using the model we just trained for futures contract extraction...\n")
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