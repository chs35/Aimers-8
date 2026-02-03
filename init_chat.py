"""
vLLM 직접 로딩 채팅
serve.py 없이 바로 실행 가능
"""
import argparse

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="EXAONE Chat")
    parser.add_argument("--model", "-m", required=True,
                        help="모델 경로 (예: models/EXAONE-4.0-1.2B)")
    args = parser.parse_args()

    print("=" * 50)
    print(f"Loading model: {args.model}")
    print("=" * 50)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    # vLLM 모델 로드 (대회 설정과 동일)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        trust_remote_code=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        max_tokens=16384,
        temperature=0.7,
        top_p=0.9,
    )

    print("=" * 50)
    print("EXAONE Chat")
    print("Type 'quit' to exit, 'clear' to reset")
    print("=" * 50)

    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Bye!")
                break
            if user_input.lower() == "clear":
                messages = []
                print("Conversation cleared.")
                continue
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            # 채팅 템플릿 적용
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 생성
            outputs = llm.generate([prompt], sampling_params)
            assistant_message = outputs[0].outputs[0].text

            messages.append({"role": "assistant", "content": assistant_message})

            print(f"AI: {assistant_message}")

        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
