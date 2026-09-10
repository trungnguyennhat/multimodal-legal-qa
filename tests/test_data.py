import json
import tempfile
import unittest
from pathlib import Path

from multimodal_legal_rag.data import read_json_list, split_train


class DataToolsTest(unittest.TestCase):
    def test_read_json_list_rejects_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json_list(path)

    def test_split_keeps_image_groups_separate(self) -> None:
        rows = [
            {
                "id": f"sample-{index}",
                "image_id": f"image-{index // 2}",
                "question": "q",
                "relevant_articles": [{"law_id": "law", "article_id": "1"}],
                "question_type": "Yes/No",
                "choices": None,
                "answer": "Đúng",
            }
            for index in range(10)
        ]
        laws = [{"id": "law", "title": "Law", "articles": [{"id": "1", "title": "A", "text": "text"}]}]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "output"
            for relative, value in {
                "dataset/train data/vlsp_2025_train.json": rows,
                "dataset/public_test data/vlsp_2025_public_test.json": rows[:2],
                "dataset/law_db/vlsp2025_law.json": laws,
            }.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            manifest = split_train(source, output, seed=2025, dev_ratio=0.2)
            self.assertEqual(manifest["image_overlap"], 0)
            self.assertEqual(manifest["sample_overlap"], 0)
            self.assertEqual(manifest["train_samples"] + manifest["dev_samples"], len(rows))


if __name__ == "__main__":
    unittest.main()
