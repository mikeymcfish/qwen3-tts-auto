import unittest

from audiobook_qwen3 import build_ffmetadata_with_chapters, chapter_start_times_from_batches


class ChapterMetadataTests(unittest.TestCase):
    def test_chapter_start_times_map_batch_numbers(self) -> None:
        chapter_times = chapter_start_times_from_batches(
            chapter_batch_numbers=[1, 3],
            part_start_samples=[0, 1200, 2600],
            sample_rate=1000,
        )
        self.assertEqual(chapter_times, [0.0, 2.6])

    def test_chapter_start_times_drop_invalid_and_duplicate_entries(self) -> None:
        chapter_times = chapter_start_times_from_batches(
            chapter_batch_numbers=[3, 3, 99, -4, 1],
            part_start_samples=[0, 1000, 2000],
            sample_rate=1000,
        )
        self.assertEqual(chapter_times, [0.0, 2.0])

    def test_ffmetadata_contains_chapter_blocks(self) -> None:
        metadata = build_ffmetadata_with_chapters(
            chapter_start_times=[0.0, 10.5],
            total_duration_seconds=20.0,
        )
        self.assertIn(";FFMETADATA1", metadata)
        self.assertIn("START=0", metadata)
        self.assertIn("END=10500", metadata)
        self.assertIn("START=10500", metadata)
        self.assertIn("END=20000", metadata)
        self.assertIn("title=Chapter 1", metadata)
        self.assertIn("title=Chapter 2", metadata)


if __name__ == "__main__":
    unittest.main()
