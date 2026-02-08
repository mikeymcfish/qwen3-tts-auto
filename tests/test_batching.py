import unittest

from audiobook_qwen3 import build_batches, split_into_paragraphs


class BatchingTests(unittest.TestCase):
    def test_split_into_paragraphs(self) -> None:
        text = "Para one line 1\nline 2\n\n\n  Para two \n\nPara three"
        self.assertEqual(
            split_into_paragraphs(text),
            ["Para one line 1 line 2", "Para two", "Para three"],
        )

    def test_batches_only_break_between_paragraphs(self) -> None:
        paragraphs = ["a" * 60, "b" * 60, "c" * 60]
        batches = build_batches(paragraphs, max_chars_per_batch=130)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].start_paragraph, 0)
        self.assertEqual(batches[0].end_paragraph, 1)
        self.assertEqual(batches[1].start_paragraph, 2)
        self.assertEqual(batches[1].end_paragraph, 2)

    def test_long_paragraph_is_single_batch(self) -> None:
        paragraphs = ["x" * 250]
        batches = build_batches(paragraphs, max_chars_per_batch=120)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].char_count, 250)


if __name__ == "__main__":
    unittest.main()
