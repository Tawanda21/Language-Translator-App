"""Simple streaming loader for Hugging Face datasets.

This script demonstrates how to import datasets from the Hub without
downloading the entire corpus to local disk. It streams examples
and outputs tokenized JSON lines to stdout so they can be piped into
training or preprocessing steps.

Usage examples:
  python data/load_datasets.py --dataset wmt14 --config de-en --split train --max-examples 1000

You can pipe the output to another script, or modify this file to integrate
with your tokenizer/training pipeline directly.
"""
import argparse
import json
import os
import sys

from datasets import load_dataset


def try_load_stream(candidates, split: str):
    """Attempt to load a streaming dataset from a list of (name, config) candidates.

    Returns the streaming dataset iterator on success, or raises the last exception.
    """
    last_exc = None
    for name, cfg in candidates:
        try:
            if cfg:
                ds = load_dataset(name, cfg, split=split, streaming=True)
            else:
                ds = load_dataset(name, split=split, streaming=True)
            return ds
        except Exception as e:
            last_exc = e
            print(f"dataset candidate failed: {name} {cfg} -> {e}", file=sys.stderr)
            continue
    raise last_exc


def stream_langpair(langpair: str, split: str, max_examples: int | None):
    """Stream a language pair like 'en-de' or 'en-af'. Yields (source, target) tuples.

    This function will try several candidate datasets for the pair and fall back
    until one succeeds. The caller should handle IO/serialization.
    """
    # mapping of language pair -> list of (dataset_name, config)
    candidates_map = {
        "en-de": [("wmt14", "de-en"), ("opus100", "en-de"), ("opus_books", "en-de"), ("opus", "en-de")],
        "en-af": [("opus100", "en-af"), ("opus_books", "en-af"), ("opus", "en-af"), ("ted_hrlr_translate", "af-en")],
        # allow reversed pairs
        "de-en": [("wmt14", "de-en"), ("opus100", "de-en"), ("opus_books", "de-en")],
        "af-en": [("opus100", "af-en"), ("opus_books", "af-en"), ("opus", "af-en")],
    }

    if langpair not in candidates_map:
        # generic fallback: try using OPUS100 with the langpair if available
        a, b = langpair.split("-")
        candidates = [("opus100", f"{a}-{b}"), ("opus", f"{a}-{b}")]
    else:
        candidates = candidates_map[langpair]

    ds = try_load_stream(candidates, split)

    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break

        src = ""
        tgt = ""

        # Heuristic: many translation datasets use a nested `translation` field
        tr = ex.get("translation") if isinstance(ex, dict) else None
        if isinstance(tr, dict):
            # choose languages from langpair
            src_lang, tgt_lang = langpair.split("-")
            # try direct keys
            src = tr.get(src_lang, "") or tr.get(src_lang.lower(), "") or tr.get(src_lang.upper(), "")
            tgt = tr.get(tgt_lang, "") or tr.get(tgt_lang.lower(), "") or tr.get(tgt_lang.upper(), "")
            if not src or not tgt:
                # try reversed keys
                keys = list(tr.keys())
                if len(keys) >= 2:
                    src = tr.get(keys[0], "")
                    tgt = tr.get(keys[1], "")
        else:
            # Generic fields
            src = ex.get("source", ex.get("src", ""))
            tgt = ex.get("target", ex.get("tgt", ""))

        if not isinstance(src, str) or not isinstance(tgt, str):
            # try to stringify
            try:
                src = str(src)
                tgt = str(tgt)
            except Exception:
                continue

        src, tgt = src.strip(), tgt.strip()
        if not src or not tgt:
            continue

        yield src, tgt


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--dataset", help="Dataset name on the Hub, e.g. wmt14 or opus")
    parser.add_argument("--config", default=None, help="Dataset config (e.g. de-en)")
    parser.add_argument("--pairs", nargs="*", help="Language pairs to stream, e.g. en-de en-af")
    parser.add_argument("--split", default="train", help="Which split to stream (train/validation/test)")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to stream (debug)")
    parser.add_argument("--out-dir", default=None, help="Optional directory to write per-pair JSONL files")
    args = parser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.pairs:
        pairs = []
        for p in args.pairs:
            # allow comma-separated single arg
            pairs.extend([x for x in p.split(",") if x])

        for pair in pairs:
            if args.out_dir:
                out_path = os.path.join(args.out_dir, f"{pair}.{args.split}.jsonl")
                with open(out_path, "w", encoding="utf-8") as fh:
                    for src, tgt in stream_langpair(pair, args.split, args.max_examples):
                        fh.write(json.dumps({"source": src, "target": tgt}, ensure_ascii=False) + "\n")
                print(f"Wrote stream for {pair} -> {out_path}", file=sys.stderr)
            else:
                # write to stdout with pair metadata
                for src, tgt in stream_langpair(pair, args.split, args.max_examples):
                    print(json.dumps({"pair": pair, "source": src, "target": tgt}, ensure_ascii=False))
    elif args.dataset:
        # legacy single-dataset streaming (uses dataset+config)
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
        for i, ex in enumerate(ds):
            if args.max_examples and i >= args.max_examples:
                break
            # Heuristic: many translation datasets use a nested `translation` field
            if isinstance(ex.get("translation"), dict):
                if args.config and "-" in args.config:
                    src_lang, tgt_lang = args.config.split("-")
                else:
                    langs = list(ex["translation"].keys())
                    src_lang, tgt_lang = langs[0], langs[1]
                src = ex["translation"].get(src_lang, "")
                tgt = ex["translation"].get(tgt_lang, "")
            else:
                src = ex.get("source", ex.get("src", ""))
                tgt = ex.get("target", ex.get("tgt", ""))

            src, tgt = (str(src).strip(), str(tgt).strip())
            if not src or not tgt:
                continue
            print(json.dumps({"source": src, "target": tgt}, ensure_ascii=False))
    else:
        parser.error("Either --pairs or --dataset must be provided. Example: --pairs en-de en-af")


if __name__ == "__main__":
    main()
