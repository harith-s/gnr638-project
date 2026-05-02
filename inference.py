import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import argparse
import gc
import re
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


VLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
TEXT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

OCR_SYSTEM = (
    "You are an expert at reading academic documents and converting them to LaTeX.\n"
    "Extract the exact content of the image faithfully."
)
OCR_USER = (
    "Convert this MCQ image to structured text exactly as follows:\n\n"
    "TITLE: <question title>\n\n"
    "QUESTION: <full question text, using LaTeX for any math e.g. $\\frac{a}{b}$>\n\n"
    "OPTIONS:\n"
    "A. <option A, using LaTeX for math, plain code for code>\n"
    "B. <option B>\n"
    "C. <option C>\n"
    "D. <option D>\n\n"
    "Preserve all mathematical notation, code, and formatting exactly."
)

ANS_SYSTEM = (
    "You are an expert in deep learning and machine learning with strong coding skills.\n"
    "You will be given a multiple choice question in LaTeX/text format.\n"
    "Reason carefully step by step and select the correct answer.\n"
    "Be especially careful with sign conventions and directionality "
    "(e.g., asymmetric quantities, gradient flow direction, position vs index ordering).\n"
    "If the question requires arithmetic, write each computation step explicitly "
    "before selecting an option.\n"
    "If two options look similar with one having a typo or off-by-one difference, "
    "double-check your computation rather than guessing."
)
ANS_USER_TEMPLATE = (
    "Here is a deep learning multiple choice question:\n\n"
    "{latex_text}\n\n"
    "Reason step by step about each option.\n"
    'On the final line write only: "Answer: X" where X is A, B, C, or D.\n'
    'If truly unsure, write "Answer: E" to skip.'
)

LETTER_MAP = {"A": 1, "B": 2, "C": 3, "D": 4}


def parse_response(response: str) -> int:
    m = re.search(r"Answer:\s*([ABCDE1-5])", response, re.IGNORECASE)
    if m:
        val = m.group(1).upper()
        if val == "E":
            return 5
        if val in LETTER_MAP:
            return LETTER_MAP[val]
        if val.isdigit():
            v = int(val)
            return v if v in {1, 2, 3, 4, 5} else 5
    letters = re.findall(r"\b([ABCD])\b", response.upper())
    if letters:
        return LETTER_MAP[letters[-1]]
    digits = re.findall(r"[1-5]", response)
    if digits:
        return int(digits[-1])
    return 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Absolute path to directory containing test.csv and images/")
    parser.add_argument("--start", type=int, default=0,
                        help="First data row index to process (inclusive, header excluded)")
    parser.add_argument("--stop", type=int, default=None,
                        help="Last data row index to process (exclusive). Defaults to len(df).")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    image_dir = test_dir / "images"
    test_csv = test_dir / "test.csv"
    output_csv = Path.cwd() / "submission.csv"

    full_df = pd.read_csv(test_csv)  # header auto-detected
    total = len(full_df)
    start = max(0, args.start)
    stop = total if args.stop is None else min(total, args.stop)
    df = full_df.iloc[start:stop].reset_index(drop=True)
    print(f"Loaded {total} test samples; processing rows [{start}:{stop}] -> {len(df)} samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.manual_seed(0)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Stage 1: OCR with VLM
    print("Loading VLM...")
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)

    def extract_latex(image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": OCR_SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": OCR_USER},
            ]},
        ]
        text = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm_processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = vlm.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return vlm_processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    latex_texts = {}
    for i, row in df.iterrows():
        img_name = row["image_name"]
        img_path = image_dir / f"{img_name}.png"
        if not img_path.exists():
            print(f"[OCR {i+1}/{len(df)}] {img_name} -- MISSING", flush=True)
            latex_texts[img_name] = None
            continue
        latex_texts[img_name] = extract_latex(img_path)
        print(f"[OCR {i+1}/{len(df)}] {img_name}", flush=True)

    del vlm, vlm_processor
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2: reasoning with text LLM
    print("Loading text LLM...")
    text_model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)

    def answer_once(latex_text: str) -> int:
        prompt = ANS_USER_TEMPLATE.format(latex_text=latex_text)
        messages = [
            {"role": "system", "content": ANS_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = text_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = text_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = text_model.generate(
                **inputs, max_new_tokens=2048, do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        response = text_tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return parse_response(response)

    results = []
    for i, row in df.iterrows():
        img_name = row["image_name"]
        latex_text = latex_texts.get(img_name)
        if latex_text is None:
            results.append({"id": img_name, "image_name": img_name, "option": 5})
        else:
            ans = answer_once(latex_text)
            print(f"[ANS {i+1}/{len(df)}] {img_name} -> {ans}", flush=True)
            results.append({"id": img_name, "image_name": img_name, "option": ans})

        # Flush partial submission every 10 samples so a walltime kill
        # leaves a usable file behind.
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            partial = pd.DataFrame(results, columns=["id", "image_name", "option"])
            partial["option"] = partial["option"].apply(lambda x: x if x in {1,2,3,4,5} else 5)
            partial.to_csv(output_csv, index=False)

    out_df = pd.DataFrame(results, columns=["id", "image_name", "option"])
    out_df["option"] = out_df["option"].apply(lambda x: x if x in {1, 2, 3, 4, 5} else 5)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
