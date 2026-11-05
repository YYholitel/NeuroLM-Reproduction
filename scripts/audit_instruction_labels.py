"""Decode the exact source-defined instruction targets for every task/class."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import tiktoken
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.author_dir / "downstream_dataset.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda text: enc.encode(text, allowed_special={"<|endoftext|>"})
    tasks = {"TUABLoader": "TUAB", "TUEVLoader": "TUEV", "SEEDDataset": "SEED",
             "HMCLoader": "HMC", "WorkloadLoader": "Workload", "TUSLLoader": "TUSL"}
    result = {"source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "scope": "all_text_templates_and_exact_label_assignments; no_EEG_or_model_execution",
              "tasks": {}}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in tasks:
            continue
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        getitem = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
        assignments = [n for n in ast.walk(init) if isinstance(n, ast.Assign)]

        def attribute_assignment(suffix):
            return next(n for n in assignments if any(
                isinstance(t, ast.Attribute) and t.attr in [suffix, "__" + suffix]
                for t in n.targets))

        text_node = attribute_assignment("text")
        prompt_node = attribute_assignment("prompt")
        # Execute only the author's template expressions and three label lines;
        # this avoids constructing fake EEG/SEED data or reimplementing slicing.
        environment = {"torch": torch, "encode": encode}
        texts = eval(compile(ast.Expression(text_node.value), str(source_path), "eval"), environment)
        prompt = eval(compile(ast.Expression(prompt_node.value), str(source_path), "eval"), environment)
        label_nodes = [n for n in getitem.body if isinstance(n, ast.Assign) and any(
            (isinstance(t, ast.Name) and t.id in ["Y_text", "prompt_len"])
            or (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "Y_text")
            for t in n.targets)]
        assert len(label_nodes) == 3
        label_code = compile(ast.Module(body=label_nodes, type_ignores=[]), str(source_path), "exec")
        obj = SimpleNamespace()
        setattr(obj, prompt_node.targets[0].attr, prompt)
        cases = []
        for label, text in texts.items():
            valid_len = text.numel()
            padded = torch.full((max(80, valid_len),), 50256, dtype=torch.long)
            padded[:valid_len] = text
            scope = {"torch": torch, "self": obj, "text": padded, "valid_text_len": valid_len}
            exec(label_code, scope)
            targets = scope["Y_text"]
            positions = torch.where(targets != -1)[0]
            target_ids = targets[positions].tolist()
            source_start = int(positions[0]) + 1
            full_text = enc.decode(text[1:].tolist())
            answer_header = full_text[:full_text.rindex("Answer:") + len("Answer:")]
            header_ids = [50257] + encode(answer_header)
            checks = {
                "prompt_is_exact_token_prefix": torch.equal(text[:prompt.numel()], prompt),
                "answer_header_is_exact_token_prefix": text[:len(header_ids)].tolist() == header_ids,
                "targets_match_next_tokens": torch.equal(targets[positions], padded[positions + 1]),
                "targets_are_after_Answer_colon": source_start >= len(header_ids),
                "target_positions_contiguous": positions.tolist() == list(range(int(positions[0]), valid_len - 1)),
                "padding_ignored": bool((targets[valid_len - 1:] == -1).all()),
                "ends_with_eot": target_ids[-1] == 50256,
            }
            assert all(checks.values()), (tasks[cls.name], label, checks)
            cases.append({"class": label, "prompt_tokens": prompt.numel(), "valid_text_tokens": valid_len,
                          "loss_positions": [int(positions[0]), int(positions[-1])],
                          "target_source_start": source_start,
                          "supervised_tokens": len(target_ids), "supervised_text": enc.decode(target_ids),
                          "prompt_tail_tokens": [{"id": int(i), "text": enc.decode([int(i)])} for i in prompt[-3:]],
                          "first_target_token": {"id": target_ids[0], "text": enc.decode([target_ids[0]])},
                          "checks": checks})
        result["tasks"][tasks[cls.name]] = {
            "label_source": [ast.get_source_segment(source, n) for n in label_nodes], "cases": cases}
        first = cases[0]
        print(tasks[cls.name], "classes=", len(cases), "prompt_tokens=", first["prompt_tokens"],
              "target_start=", first["target_source_start"], "supervised=", repr(first["supervised_text"]))
    assert len(result["tasks"]) == 6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("INSTRUCTION_LABEL_AUDIT_OK")


if __name__ == "__main__":
    main()
