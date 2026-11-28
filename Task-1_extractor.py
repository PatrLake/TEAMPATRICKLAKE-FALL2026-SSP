from pathlib import Path
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import sys
import torch
import yaml


def is_valid_input(filepath: str) -> bool:
    try:
        path = Path(filepath)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("rb") as f:
            if not f.read(5).startswith(b"%PDF-"):
                return False
        reader = PdfReader(path, strict=False)
        if reader.is_encrypted or len(reader.pages) == 0:
            return False
        _ = reader.pages[0]
    except Exception:
        return False
    return True


def construct_zero_shot_prompt(doc_text: str) -> str:
    return (
        "Identify all 7 top-level Key Data Elements (sections 1 through 7) and up to 5 key "
        "security requirements for each section from the CIS Docker Benchmark below.\n"
        "Do not repeat requirements. Output ONLY valid YAML with double-quoted strings:\n"
        "element1:\n"
        '  name: "<section name>"\n'
        "  requirements:\n"
        '    - "<req1>"\n'
        '    - "<req2>"\n\n'
        f"{doc_text}"
    )


def construct_few_shot_prompt(doc_text: str) -> str:
    return (
        "Identify all 7 top-level Key Data Elements (sections 1 through 7) and up to 5 key "
        "security requirements for each section from the CIS Docker Benchmark below.\n"
        "Do not repeat requirements. Output ONLY valid YAML with double-quoted strings.\n\n"
        "Example Input:\n"
        "1 Network Access\n"
        "1.1 Disable unused network ports (Automated)\n"
        "1.2 Require TLS for remote connections (Manual)\n\n"
        "Example Output:\n"
        "```yaml\n"
        "element1:\n"
        '  name: "1 Network Access"\n'
        "  requirements:\n"
        '    - "1.1 Disable unused network ports (Automated)"\n'
        '    - "1.2 Require TLS for remote connections (Manual)"\n'
        "```\n\n"
        "Now do the same for all 7 sections in the document below:\n"
        f"{doc_text}"
    )


def construct_chain_of_thought_prompt(doc_text: str) -> str:
    return (
        "Identify all 7 top-level Key Data Elements (sections 1 through 7) and up to 5 key "
        "security requirements for each section from the CIS Docker Benchmark below.\n"
        "Follow these steps:\n"
        "1. Briefly list the 7 top-level section headings as Key Data Elements.\n"
        "2. Select up to 5 distinct numbered CIS recommendations for each section without repeating any.\n"
        "3. Output the final nested dictionary inside a single ```yaml block with double-quoted strings:\n"
        "```yaml\n"
        "element1:\n"
        '  name: "<section name>"\n'
        "  requirements:\n"
        '    - "<req1>"\n'
        '    - "<req2>"\n'
        "```\n\n"
        f"{doc_text}"
    )


def identify_KDEs(prompt: str, filepath: str, doc_num: int, prompt_type: str, tokenizer, model) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=False,
            repetition_penalty=1.15,
        )

    raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    blocks = re.findall(r"```(?:ya?ml)?\s*(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
    if blocks:
        yaml_text = blocks[-1].strip()
    else:
        yaml_text = re.sub(r"^```(?:ya?ml)?\s*", "", raw_output, flags=re.IGNORECASE).strip()

    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        try:
            parsed = yaml.safe_load("\n".join(yaml_text.splitlines()[:-1]))
        except yaml.YAMLError:
            parsed = yaml_text

    yaml_path = Path(f"Task-1_{Path(filepath).stem}_doc{doc_num}-kdes.yaml")
    data = {}
    if prompt_type != "zero-shot" and yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    data[prompt_type] = parsed

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return raw_output


def create_TXT_file(prompts: list, outputs: list) -> None:
    llm_name = "google/gemma-3-1b-it"
    prompt_types = ["zero-shot", "few-shot", "chain-of-thought"]
    with open("Task-1_llm_outputs.txt", "w", encoding="utf-8") as f:
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            prompt_type = prompt_types[i % 3]
            f.write("*LLM Name*\n")
            f.write(f"{llm_name}\n")
            f.write("*Prompt Used*\n")
            f.write(f"{prompt}\n")
            f.write("*Prompt Type*\n")
            f.write(f"{prompt_type}\n")
            f.write("*LLM Output*\n")
            f.write(f"{output}\n\n")


if __name__ == "__main__":
    filepath1 = sys.argv[1]
    filepath2 = sys.argv[2]

    if not (is_valid_input(filepath1) and is_valid_input(filepath2)):
        sys.exit(1)

    model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompts = []
    outputs = []

    for doc_num, filepath in enumerate((filepath1, filepath2), start=1):
        reader = PdfReader(filepath, strict=False)
        pages = []
        in_summary = False
        for page in reader.pages:
            text = page.extract_text() or ""
            if not in_summary:
                if (
                    re.search(r"Appendix:\s+Summary\s+Table", text, re.IGNORECASE)
                    and re.search(r"CIS\s+Benchmark\s+Recommendation", text)
                ):
                    in_summary = True
                else:
                    continue
            elif re.search(r"Appendix:\s+CIS\s+Controls", text, re.IGNORECASE):
                break
            pages.append(text)

        if not pages:
            pages = [page.extract_text() or "" for page in reader.pages]

        lines = []
        for page_text in pages:
            for line in page_text.splitlines():
                line = re.sub(r"[\u25a1]+", "", line).strip()
                line = re.sub(r"(^|\s+)o\s+o$", "", line).strip()
                if not line or re.match(
                    r"^(Page \d+|Internal Only - General|Appendix: Summary Table|CIS Benchmark Recommendation.*|Set|Correctly|Yes No)$",
                    line,
                ):
                    continue
                lines.append(line)
        doc_text = "\n".join(lines)

        for prompt_type, func in (
            ("zero-shot", construct_zero_shot_prompt),
            ("few-shot", construct_few_shot_prompt),
            ("chain-of-thought", construct_chain_of_thought_prompt),
        ):
            prompt = func(doc_text)
            output = identify_KDEs(prompt, filepath, doc_num, prompt_type, tokenizer, model)
            prompts.append(prompt)
            outputs.append(output)

    create_TXT_file(prompts, outputs)

    prompt_types = ["zero-shot", "few-shot", "chain-of-thought"]
    with open("PROMPT.md", "w", encoding="utf-8") as f:
        for idx, p_type in enumerate(prompt_types):
            f.write(f"## {p_type}\n\n")
            for doc_idx in range(idx, len(prompts), 3):
                f.write(f"### Document {doc_idx // 3 + 1}\n\n")
                f.write("````text\n")
                f.write(f"{prompts[doc_idx]}\n")
                f.write("````\n\n")