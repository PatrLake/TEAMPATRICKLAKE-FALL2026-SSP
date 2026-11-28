import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from pypdf import PdfWriter
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
extractor = importlib.import_module("Task-1_extractor")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, token_ids, skip_special_tokens=True):
        return (
            "```yaml\n"
            "element1:\n"
            '  name: "1 Host Configuration"\n'
            "  requirements:\n"
            '    - "1.1.1 Ensure a separate partition (Manual)"\n'
            "```"
        )


class FakeModel:
    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


class TestTask1Extractor(unittest.TestCase):
    def test_is_valid_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_pdf = Path(tmpdir) / "valid.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            with valid_pdf.open("wb") as f:
                writer.write(f)

            invalid_file = Path(tmpdir) / "not_a_pdf.txt"
            invalid_file.write_text("not a pdf", encoding="utf-8")

            self.assertTrue(extractor.is_valid_input(str(valid_pdf)))
            self.assertFalse(extractor.is_valid_input(str(invalid_file)))
            self.assertFalse(extractor.is_valid_input(str(Path(tmpdir) / "missing.pdf")))

    def test_construct_zero_shot_prompt(self):
        sample = "1 Host Configuration\n1.1.1 Ensure a separate partition (Manual)"
        prompt = extractor.construct_zero_shot_prompt(sample)
        self.assertIsInstance(prompt, str)
        self.assertIn(sample, prompt)
        self.assertIn("YAML", prompt)

    def test_construct_few_shot_prompt(self):
        sample = "1 Host Configuration\n1.1.1 Ensure a separate partition (Manual)"
        prompt = extractor.construct_few_shot_prompt(sample)
        self.assertIsInstance(prompt, str)
        self.assertIn(sample, prompt)
        self.assertIn("Example Input:", prompt)
        self.assertIn("Example Output:", prompt)

    def test_construct_chain_of_thought_prompt(self):
        sample = "1 Host Configuration\n1.1.1 Ensure a separate partition (Manual)"
        prompt = extractor.construct_chain_of_thought_prompt(sample)
        self.assertIsInstance(prompt, str)
        self.assertIn(sample, prompt)
        self.assertIn("Follow these steps:", prompt)

    def test_identify_KDEs(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                output = extractor.identify_KDEs(
                    "test prompt",
                    "sample.pdf",
                    1,
                    "zero-shot",
                    FakeTokenizer(),
                    FakeModel(),
                )
                self.assertIsInstance(output, str)
                yaml_path = Path("Task-1_sample_doc1-kdes.yaml")
                self.assertTrue(yaml_path.is_file())
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                self.assertEqual(data["zero-shot"]["element1"]["name"], "1 Host Configuration")
            finally:
                os.chdir(old_cwd)

    def test_create_TXT_file(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                extractor.create_TXT_file(["p1", "p2", "p3"], ["o1", "o2", "o3"])
                txt_path = Path("Task-1_llm_outputs.txt")
                self.assertTrue(txt_path.is_file())
                content = txt_path.read_text(encoding="utf-8")
                self.assertIn("*LLM Name*\ngoogle/gemma-3-1b-it", content)
                self.assertIn("*Prompt Used*\np1", content)
                self.assertIn("*Prompt Type*\nzero-shot", content)
                self.assertIn("*Prompt Type*\nfew-shot", content)
                self.assertIn("*Prompt Type*\nchain-of-thought", content)
                self.assertIn("*LLM Output*\no3", content)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()