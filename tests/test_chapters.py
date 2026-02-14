import unittest

from audiobook_qwen3 import (
    build_batch_boundary_types,
    build_ffmetadata_with_chapters,
    chapter_start_times_from_batches,
    compute_inter_batch_pause_samples,
)


class ChapterMetadataTests(unittest.TestCase):
    def test_build_batch_boundary_types_marks_chapters(self) -> None:
        boundary_types = build_batch_boundary_types(5, [3, 5])
        self.assertEqual(
            boundary_types,
            ["none", "natural", "chapter", "natural", "chapter"],
        )

    def test_compute_inter_batch_pause_samples_adds_chapter_pause(self) -> None:
        gap = compute_inter_batch_pause_samples(
            base_pause_samples=300,
            chapter_pause_samples=900,
            next_batch_number=4,
            chapter_batch_numbers={2, 4},
        )
        self.assertEqual(gap, 1200)

    def test_compute_inter_batch_pause_samples_without_chapter(self) -> None:
        gap = compute_inter_batch_pause_samples(
            base_pause_samples=300,
            chapter_pause_samples=900,
            next_batch_number=3,
            chapter_batch_numbers={2, 4},
        )
        self.assertEqual(gap, 300)

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
