import unittest

from audiobook_qwen3 import format_plain_progress


class PlainProgressFormattingTests(unittest.TestCase):
    def test_format_plain_progress_includes_percent_elapsed_and_eta(self) -> None:
        line = format_plain_progress(
            completed_batches=2,
            total_batches=8,
            completed_chars=1000,
            total_chars=4000,
            elapsed_seconds=125,
            average_batch_seconds=10,
        )
        self.assertIn("batches 2/8 ( 25.0%)", line)
        self.assertIn("chars 1000/4000 ( 25.0%)", line)
        self.assertIn("elapsed 00:02:05", line)
        self.assertIn("eta 00:01:00", line)

    def test_format_plain_progress_handles_zero_totals(self) -> None:
        line = format_plain_progress(
            completed_batches=0,
            total_batches=0,
            completed_chars=0,
            total_chars=0,
            elapsed_seconds=0,
            average_batch_seconds=0,
        )
        self.assertIn("batches 0/1 (  0.0%)", line)
        self.assertIn("chars 0/1 (  0.0%)", line)
        self.assertIn("elapsed 00:00:00", line)
        self.assertIn("eta 00:00:00", line)


if __name__ == "__main__":
    unittest.main()
