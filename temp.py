from datasets import load_from_disk
from transformers import AutoTokenizer

ds = load_from_disk('/workspace/data/calibration/dataset')
tokenizer = AutoTokenizer.from_pretrained('/workspace/models/EXAONE-4.0-1.2B', trust_remote_code=True)

for i in range(3):
    ex = ds[i]
    text = tokenizer.apply_chat_template(
        ex['messages'],
        tokenize=False,
        add_generation_prompt=False,
    )
    print(f'=== [{ex["source"]}] source_id={ex["source_id"]} ===')
    print(text)
    print()
