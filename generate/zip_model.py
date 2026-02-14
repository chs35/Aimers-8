"""
models/ 폴더 안의 특정 모델 폴더를 .zip으로 압축하는 스크립트.

사용법:
    python zip_model.py                          # 대화형으로 모델 선택
    python zip_model.py <모델폴더명>              # 직접 지정
    python zip_model.py EXAONE-4.0-1.2B-Pruned   # 예시
"""

import os
import sys
import zipfile
from pathlib import Path

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def list_model_dirs(models_dir: str) -> list[str]:
    """models/ 안의 하위 폴더 목록 반환."""
    if not os.path.isdir(models_dir):
        return []
    return sorted(
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    )


def zip_model(model_dir: str, output_path: str) -> None:
    """model_dir 전체를 output_path(.zip)로 압축."""
    model_dir = os.path.abspath(model_dir)
    base = os.path.basename(model_dir)
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(model_dir)
        for f in filenames
    )
    print(f"압축 대상: {model_dir}")
    print(f"총 크기:   {total_size / (1024**3):.2f} GB")
    print(f"출력 파일: {output_path}")
    print()

    written = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        for dirpath, _, filenames in os.walk(model_dir):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                arcname = os.path.join("model", os.path.relpath(full, model_dir))
                fsize = os.path.getsize(full)
                written += fsize
                pct = written / total_size * 100 if total_size else 100
                print(f"  [{pct:5.1f}%] {arcname} ({fsize / (1024**2):.1f} MB)")
                zf.write(full, arcname)

    final_size = os.path.getsize(output_path)
    print()
    print(f"완료! {output_path} ({final_size / (1024**3):.2f} GB)")


def main():
    models_dir = os.path.abspath(MODELS_DIR)

    # 인자로 모델명 지정
    if len(sys.argv) >= 2:
        model_name = sys.argv[1]
    else:
        dirs = list_model_dirs(models_dir)
        if not dirs:
            print(f"models/ 폴더에 모델이 없습니다: {models_dir}")
            sys.exit(1)

        print("=== 사용 가능한 모델 폴더 ===")
        for i, d in enumerate(dirs, 1):
            print(f"  [{i}] {d}")
        print()

        choice = input("압축할 모델 번호를 선택하세요: ").strip()
        try:
            idx = int(choice) - 1
            model_name = dirs[idx]
        except (ValueError, IndexError):
            print("잘못된 선택입니다.")
            sys.exit(1)

    model_path = os.path.join(models_dir, model_name)
    if not os.path.isdir(model_path):
        print(f"모델 폴더를 찾을 수 없습니다: {model_path}")
        sys.exit(1)

    project_root = os.path.join(models_dir, "..")
    output_path = os.path.join(project_root, "submit.zip")
    zip_model(model_path, output_path)


if __name__ == "__main__":
    main()
