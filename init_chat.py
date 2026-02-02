from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "/workspace/EXAONE-4.0-1.2B"

# 평가 서버와 동일한 설정
print("Loading model (eval server config)...")
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    enforce_eager=True,  # segfault 방지
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
print("Model loaded!\n")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=16384,  # max_gen_toks = 16384
)

print("=" * 50)
print("EXAONE-4.0-1.2B Chat (type 'quit' to exit)")
print("=" * 50)

while True:
    try:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Bye!")
            break
        if not user_input:
            continue

        # apply_chat_template = true (평가 서버 설정)
        messages = [{"role": "user", "content": user_input}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text
        print(f"AI: {response}")
    except KeyboardInterrupt:
        print("\nBye!")
        break
