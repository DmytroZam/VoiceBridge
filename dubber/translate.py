"""Машинный перевод — Meta NLLB-200 (200 языков, open-source)."""
from __future__ import annotations

import re
from typing import Callable

from .utils import free_gpu, setup_cuda_dlls

_SENT_RE = re.compile(r"(?<=[.!?…。！？])\s+")
_CT2_MODELS = {
    "facebook/nllb-200-distilled-1.3B": "Derur/nllb-200-distilled-1.3B-ct2-float16",
}


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def restore_punctuation(src: str, tgt: str) -> str:
    """Переносит «!» и «?» из оригинала — NLLB их часто теряет, а XTTS по ним строит интонацию."""
    s, t = src.rstrip(), tgt.rstrip()
    if not s or not t:
        return tgt
    tail = "".join(ch for ch in s[-3:] if ch in "!?") or ""
    if not tail:
        return tgt
    core = t.rstrip(".…!?")
    return (core + tail[-2:]) if core else tgt


def translate(texts: list[str], src_nllb: str, tgt_nllb: str,
              model_name: str = "facebook/nllb-200-distilled-1.3B", device: str = "cuda",
              batch_size: int = 8, progress: Callable[[float, str], None] | None = None) -> list[str]:
    """Переводит список фраз. Каждая фраза делится на предложения: NLLB часто
    «теряет» часть текста, если подать ему несколько предложений сразу."""
    if not texts:
        return []
    if src_nllb == tgt_nllb:
        return list(texts)
    sentences, owner = [], []
    for i, t in enumerate(texts):
        for s in split_sentences(t):
            sentences.append(s)
            owner.append(i)
    translated = _translate_raw(sentences, src_nllb, tgt_nllb, model_name, device, batch_size, progress)
    translated = [restore_punctuation(s, t) for s, t in zip(sentences, translated)]
    joined = [[] for _ in texts]
    for i, s in zip(owner, translated):
        joined[i].append(s)
    return [" ".join(p) for p in joined]


def _load_ct2_dir(ct2_repo: str) -> str:
    """Возвращает путь к кэшированной CT2-модели без сетевых обращений и WinError 1314 при наличии кэша."""
    import os
    from huggingface_hub import file_download, snapshot_download

    try:
        return snapshot_download(ct2_repo, local_files_only=True)
    except Exception:
        if os.name == "nt":
            file_download.are_symlinks_supported = lambda *_a, **_kw: False
        return snapshot_download(ct2_repo)


def _load_tokenizer(model_name: str, src_nllb: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, src_lang=src_nllb, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, src_lang=src_nllb)


def _translate_ct2(texts, src_nllb, tgt_nllb, tok_model, ct2_repo, device, batch_size, progress) -> list[str]:
    """Быстрый перевод через CTranslate2 FP16 без загрузки 5.48 ГБ FP32-файла PyTorch."""
    import ctranslate2

    setup_cuda_dlls(update_path=True)
    ct2_dir = _load_ct2_dir(ct2_repo)
    tok = _load_tokenizer(tok_model, src_nllb)
    tok.src_lang = src_nllb

    compute_type = "float16" if device == "cuda" else "int8"
    try:
        translator = ctranslate2.Translator(ct2_dir, device=device, compute_type=compute_type)
    except Exception:
        if device != "cuda":
            raise
        translator = ctranslate2.Translator(ct2_dir, device="cpu", compute_type="int8")

    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        src_tokens = [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=512)) for t in batch]
        results = translator.translate_batch(
            src_tokens,
            target_prefix=[[tgt_nllb]] * len(src_tokens),
            beam_size=4,
            no_repeat_ngram_size=4,
            max_decoding_length=512,
        )
        for res in results:
            hyp = res.hypotheses[0] if res.hypotheses else []
            if hyp and hyp[0] == tgt_nllb:
                hyp = hyp[1:]
            out.append(tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True).strip())
        if progress:
            done = min(i + batch_size, len(texts))
            progress(done / len(texts), f"Translated {done}/{len(texts)}")

    del translator, tok
    free_gpu()
    return out


def _translate_raw(texts, src_nllb, tgt_nllb, model_name, device, batch_size, progress) -> list[str]:
    ct2_repo = _CT2_MODELS.get(model_name)
    if ct2_repo:
        return _translate_ct2(texts, src_nllb, tgt_nllb, model_name, ct2_repo, device, batch_size, progress)

    import os

    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
    import torch
    from transformers import AutoModelForSeq2SeqLM

    setup_cuda_dlls()
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = _load_tokenizer(model_name, src_nllb)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    bos = tok.convert_tokens_to_ids(tgt_nllb)

    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                forced_bos_token_id=bos,
                num_beams=4,
                max_new_tokens=int(enc["input_ids"].shape[1] * 2.5) + 16,
                max_length=None,
                no_repeat_ngram_size=4,
            )
        out.extend(t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True))
        if progress:
            done = min(i + batch_size, len(texts))
            progress(done / len(texts), f"Translated {done}/{len(texts)}")

    del model, tok
    free_gpu()
    return out


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        src_texts = json.load(f)

    def _emit(frac: float, msg: str = ""):
        print(f"__PROG__:{frac:.4f}:{msg}", flush=True)

    res = translate(src_texts, args.src, args.tgt, args.model, args.device, progress=_emit)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
