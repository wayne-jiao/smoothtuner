import torch
from models import build_or_load_gen_model
from utils import SimpleClsDataset
import argparse

def test_model(checkpoint_path):
    # Create minimal args object
    class Args:
        pass
    
    args = Args()
    args.load_model_path = checkpoint_path
    args.model_type = "codet5"
    args.model_name_or_path = "Salesforce/codet5-base"
    args.tokenizer_path = None
    args.config_name = None
    args.max_source_length = 512
    args.max_target_length = 128
    args.add_lang_ids = False
    args.from_scratch = False
    args.local_rank = 0
    args.raw_input = True
    
    # Load model
    config, model, tokenizer = build_or_load_gen_model(args)
    model = model.cuda()
    model.eval()
    
    # Test cases
    test_cases = [
        {
            "oldf": "def add(a, b):\n    return a + b",
            "patch": "@@ -1 +1 @@\n-    return a + b\n+    return a - b",
            "msg": "Bug fix: change addition to subtraction",
            "id": "test1",
            "y": 1
        },
    ]
    
    for test_input in test_cases:
        example = SimpleClsDataset.Example(
            input=test_input["oldf"],
            patch=test_input["patch"],
            msg=test_input["msg"],
            id=test_input["id"],
            y=test_input["y"]
        )
        
        tokenized = SimpleClsDataset.tokenize_example(example, tokenizer, args)
        
        with torch.no_grad():
            source_ids = torch.tensor([tokenized.source_ids], dtype=torch.long).cuda()
            source_mask = source_ids.ne(tokenizer.pad_id)
            logits = model(cls=True, input_ids=source_ids, labels=None, attention_mask=source_mask)
            prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
        print(f"Input: {test_input['msg']}")
        print(f"Prediction: {prediction}, Probabilities: {probs}")
        print("-" * 50)

if __name__ == "__main__":
    test_model("saved_models/checkpoints-last")
