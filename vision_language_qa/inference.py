from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm.part_b_model import PartBVLM


DEFAULT_CONFIG: dict[str, Any] = {
    "qwen_model": "Qwen/Qwen3-4B-Instruct-2507",
    "local_files_only": True,
    "vision_bundle": "clip_vision_encoder_best.pt",
    "projector": "projector.pt",
    "lora_adapter": "lora_adapter",
    "image_size": 224,
    "vision_token_dim": 384,
    "llm_hidden_size": 2560,
    "projector_hidden_size": 2560,
    "projector_dropout": 0.0,
    "max_seq_len": 768,
    "max_new_tokens": 128,
    "batch_size": 1,
    "inference_strategy": "generate",
    "merge_lora": False,
    "candidate_prompt_template": "Answer the CLEVR question with only the final answer.\nQuestion: {question}\nAnswer:",
    "candidate_answers": [
        "yes",
        "no",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "gray",
        "red",
        "blue",
        "green",
        "brown",
        "purple",
        "cyan",
        "yellow",
        "cube",
        "sphere",
        "cylinder",
        "small",
        "large",
        "rubber",
        "metal",
    ],
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "qa_prompt_template": (
        "Answer the question about the CLEVR image. Provide reasoning, then write the final answer "
        "after 'Answer:'.\nQuestion: {question}"
    ),
}

ANSWER_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
REASONING_PREFIX_RE = re.compile(r"^\s*reasoning\s*:\s*", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision-language question answering inference")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        user_config = load_json(config_path)
        config.update(user_config)
    config["normalization"] = {
        **DEFAULT_CONFIG["normalization"],
        **config.get("normalization", {}),
    }
    return config


def resolve_model_path(model_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return model_dir / path


def records_from_questions_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        questions = payload.get("questions")
        if isinstance(questions, list):
            return [item for item in questions if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Expected a CLEVR questions JSON object with a 'questions' list")


def resize_short_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width < height:
        new_width = size
        new_height = int(round(size * height / width))
    else:
        new_height = size
        new_width = int(round(size * width / height))
    return image.resize((new_width, new_height), Image.BICUBIC)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, int(round((width - size) / 2.0)))
    top = max(0, int(round((height - size) / 2.0)))
    return image.crop((left, top, left + size, top + size))


def preprocess_image(path: str | Path, config: dict[str, Any]) -> torch.Tensor:
    image_size = int(config.get("image_size", 224))
    image = Image.open(path).convert("RGB")
    image = resize_short_side(image, image_size + 32)
    image = center_crop(image, image_size)
    buffer = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = buffer.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def precision_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(config: dict[str, Any], model_dir: Path, device: torch.device) -> tuple[Any, PartBVLM]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qwen_source = config["qwen_model"]
    tokenizer_source = model_dir / "qwen_tokenizer"
    if not tokenizer_source.exists():
        tokenizer_source = Path(qwen_source)

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=True,
        local_files_only=bool(config.get("local_files_only", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = precision_dtype()
    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": bool(config.get("local_files_only", True)),
    }
    if dtype != torch.float32:
        model_kwargs["torch_dtype"] = dtype
    llm = AutoModelForCausalLM.from_pretrained(qwen_source, **model_kwargs)
    lora_path = resolve_model_path(model_dir, config["lora_adapter"])
    llm = PeftModel.from_pretrained(llm, lora_path)
    if bool(config.get("merge_lora", False)) and hasattr(llm, "merge_and_unload"):
        llm = llm.merge_and_unload()
    llm.to(device)
    llm.eval()
    if hasattr(llm, "config"):
        llm.config.use_cache = True

    vlm = PartBVLM(
        vision_bundle_path=str(resolve_model_path(model_dir, config["vision_bundle"])),
        llm=llm,
        vision_token_dim=int(config.get("vision_token_dim", 384)),
        llm_hidden_size=int(config.get("llm_hidden_size", 2560)),
        projector_hidden_size=int(config.get("projector_hidden_size", 2560)),
        projector_dropout=float(config.get("projector_dropout", 0.0)),
        freeze_vision=True,
    )
    projector_checkpoint = torch.load(
        resolve_model_path(model_dir, config["projector"]),
        map_location="cpu",
        weights_only=False,
    )
    projector_state = projector_checkpoint.get("projector", projector_checkpoint)
    vlm.projector.load_state_dict(projector_state)
    vlm.to(device)
    vlm.eval()
    return tokenizer, vlm


def clean_answer(answer: str) -> str:
    answer = answer.strip().splitlines()[0] if answer.strip() else ""
    answer = answer.strip().strip(string.whitespace + ".;,:")
    return " ".join(answer.lower().split())


def split_reasoning_and_answer(text: str) -> tuple[str, str]:
    text = text.strip()
    match = ANSWER_RE.search(text)
    if match is None:
        answer = clean_answer(text)
        reasoning = REASONING_PREFIX_RE.sub("", text).strip()
        return reasoning, answer

    reasoning = text[: match.start()].strip()
    reasoning = REASONING_PREFIX_RE.sub("", reasoning).strip()
    answer = clean_answer(match.group(1))
    if not reasoning:
        reasoning = text.strip()
    return reasoning, answer


def make_prompt(question: str, config: dict[str, Any]) -> str:
    return str(config["qa_prompt_template"]).format(question=question)


def make_candidate_prompt(question: str, config: dict[str, Any]) -> str:
    return str(config["candidate_prompt_template"]).format(question=question)


def candidate_token_ids(tokenizer: Any, candidates: list[str]) -> tuple[list[str], torch.Tensor]:
    kept_candidates: list[str] = []
    kept_ids: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        token_ids = tokenizer(str(candidate), add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        first_id = int(token_ids[0])
        if first_id in seen:
            continue
        seen.add(first_id)
        kept_candidates.append(str(candidate))
        kept_ids.append(first_id)
    if not kept_ids:
        raise ValueError("No usable candidate answer token ids")
    return kept_candidates, torch.tensor(kept_ids, dtype=torch.long)


@torch.no_grad()
def run_candidate_inference(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: PartBVLM,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    batch_size = max(1, int(config.get("batch_size", 1)))
    max_seq_len = int(config.get("max_seq_len", 768))
    candidates, candidate_ids = candidate_token_ids(tokenizer, list(config["candidate_answers"]))
    candidate_ids = candidate_ids.to(device)

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = torch.stack([preprocess_image(item["image_path"], config) for item in batch], dim=0).to(device)
        prompts = [make_candidate_prompt(str(item["question"]), config) for item in batch]
        prepared = model.prepare_inputs(
            images=images,
            prompts=prompts,
            targets=None,
            tokenizer=tokenizer,
            max_length=max_seq_len,
        )
        result = model.llm(inputs_embeds=prepared.inputs_embeds, attention_mask=prepared.attention_mask)
        lengths = prepared.attention_mask.sum(dim=1).sub(1).clamp_min(0)
        row_ids = torch.arange(prepared.attention_mask.size(0), device=device)
        next_token_logits = result.logits[row_ids, lengths]
        candidate_scores = next_token_logits.index_select(dim=1, index=candidate_ids)
        best_indices = candidate_scores.argmax(dim=1).tolist()
        for offset, (record, best_index) in enumerate(zip(batch, best_indices)):
            key = record.get("question_index")
            if key is None:
                key = start + offset
            answer = candidates[int(best_index)]
            outputs[str(key)] = {
                "reasoning": "Scored the CLEVR answer candidates and selected the highest-probability answer.",
                "answer": answer,
            }
    return outputs


@torch.no_grad()
def run_inference(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: PartBVLM,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    batch_size = max(1, int(config.get("batch_size", 1)))
    max_seq_len = int(config.get("max_seq_len", 768))
    max_new_tokens = int(config.get("max_new_tokens", 128))

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = torch.stack([preprocess_image(item["image_path"], config) for item in batch], dim=0).to(device)
        prompts = [make_prompt(str(item["question"]), config) for item in batch]
        generated = model.generate_text(
            images=images,
            prompts=prompts,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            max_new_tokens=max_new_tokens,
        )
        for offset, (record, text) in enumerate(zip(batch, generated)):
            key = record.get("question_index")
            if key is None:
                key = start + offset
            reasoning, answer = split_reasoning_and_answer(text)
            outputs[str(key)] = {
                "reasoning": str(reasoning),
                "answer": str(answer),
            }
    return outputs


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    config = load_config(model_dir)
    records = records_from_questions_json(load_json(args.data_path))
    missing = [idx for idx, item in enumerate(records) if not item.get("image_path")]
    if missing:
        raise ValueError("Every question must include an absolute image_path field")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(config, model_dir, device)
    if str(config.get("inference_strategy", "generate")).lower() == "candidate":
        predictions = run_candidate_inference(records, tokenizer, model, config, device)
    else:
        predictions = run_inference(records, tokenizer, model, config, device)
    save_json(args.output_file, predictions)


if __name__ == "__main__":
    main()
