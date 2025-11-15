import torch
from models import build_or_load_gen_model
from utils import ReviewExample
from configs import add_args
import argparse

def test_model(checkpoint_path):
    # Use add_args like the main script does
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    
    # Override only what's needed for testing
    args.load_model_path = checkpoint_path
    args.output_dir = checkpoint_path
    args.local_rank = 0
    
    # Load model exactly like main() does
    config, model, tokenizer = build_or_load_gen_model(args)
    model = model.cuda()
    model.eval()
    
    # Test input
    test_input = {
        "oldf": "def add(a, b):\n    return a + b",
        "patch": "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b",
        "msg": "Bug fix",
        "id": "test1",
        "y": 1
    }
    
    # Create ReviewExample like the dataset does
    example = ReviewExample(
        idx=0,
        oldf=test_input["oldf"],
        diff=test_input["patch"],
        msg=test_input.get("msg", ""),
        cmtid=test_input.get("id", ""),
        max_len=200,
        y=test_input["y"]
    )
    
    if not example.avail:
        print("Invalid diff format")
        return
    
    # Process like SimpleClsDataset.convert_examples_to_features does
    example.input_lines = example.input.split("<e0>")
    example.input_lines = example.input_lines[:len(example.labels)]
    for i in range(len(example.input_lines)):
        if example.labels[i] == 1:
            example.input_lines[i] = "+ " + example.input_lines[i]
        elif example.labels[i] == 0:
            example.input_lines[i] = "- " + example.input_lines[i]
    example.input = " ".join(example.input_lines)
    
    # Encode
    text = tokenizer.encode(example.input, max_length=args.max_source_length, truncation=True)
    # Remove BOS/EOS based on tokenizer type
    from transformers import T5Tokenizer, RobertaTokenizer
    if type(tokenizer) == T5Tokenizer:
        input_ids = text[:-1]
    elif type(tokenizer) == RobertaTokenizer:
        input_ids = text[1:-1]
    else:
        input_ids = text
    
    # Truncate if needed
    exceed_l = len(input_ids) - args.max_source_length + 2
    if exceed_l > 0:
        halfexl = (exceed_l + 1) // 2
        input_ids = input_ids[halfexl:-halfexl]
    
    source_ids = input_ids[:args.max_source_length - 2]
    source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
    pad_len = args.max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_id] * pad_len
    
    # Inference
    with torch.no_grad():
        source_ids = torch.tensor([source_ids], dtype=torch.long).cuda()
        source_mask = source_ids.ne(tokenizer.pad_id)
        logits = model(cls=True, input_ids=source_ids, labels=None, attention_mask=source_mask)
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
    print(f"Prediction: {prediction}, Probabilities: {probs}")

if __name__ == "__main__":
    test_model("saved_models/checkpoints-last")