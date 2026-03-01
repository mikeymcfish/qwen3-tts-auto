import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audiobook_qwen3 import (
    build_batch_boundary_types,
    build_batches,
    build_ffmetadata_with_chapters,
    chapter_time_entries_from_batches,
    chapter_start_times_from_batches,
    compute_inter_batch_pause_samples,
    create_continue_assets,
    extract_chapter_titles_from_raw_text,
    order_part_paths_by_batch_number,
    split_into_paragraphs,
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

    def test_chapter_time_entries_include_batch_numbers(self) -> None:
        chapter_entries = chapter_time_entries_from_batches(
            chapter_batch_numbers=[2, 4],
            part_start_samples=[0, 1000, 2500, 4000],
            sample_rate=1000,
        )
        self.assertEqual(chapter_entries, [(1.0, 2), (4.0, 4)])

    def test_chapter_time_entries_support_explicit_part_batch_numbers(self) -> None:
        chapter_entries = chapter_time_entries_from_batches(
            chapter_batch_numbers=[2, 4],
            part_start_samples=[0, 1800, 3200],
            part_batch_numbers=[1, 2, 4],
            sample_rate=1000,
        )
        self.assertEqual(chapter_entries, [(1.8, 2), (3.2, 4)])

    def test_order_part_paths_by_batch_number_sorts_when_pattern_matches(self) -> None:
        paths = [
            Path("parts") / "batch_00003.wav",
            Path("parts") / "batch_00001.wav",
            Path("parts") / "batch_00002.wav",
        ]
        ordered_paths, ordered_numbers = order_part_paths_by_batch_number(paths)
        self.assertEqual([p.name for p in ordered_paths], ["batch_00001.wav", "batch_00002.wav", "batch_00003.wav"])
        self.assertEqual(ordered_numbers, [1, 2, 3])

    def test_ffmetadata_contains_chapter_blocks(self) -> None:
        metadata = build_ffmetadata_with_chapters(
            chapter_entries=[(0.0, "Prologue"), (10.5, None)],
            total_duration_seconds=20.0,
        )
        self.assertIn(";FFMETADATA1", metadata)
        self.assertIn("START=0", metadata)
        self.assertIn("END=10500", metadata)
        self.assertIn("START=10500", metadata)
        self.assertIn("END=20000", metadata)
        self.assertIn("title=Prologue", metadata)
        self.assertIn("title=Chapter 2", metadata)

    def test_create_continue_assets_does_not_duplicate_inline_chapter_title(self) -> None:
        source_text = (
            "[CHAPTER] Prologue\n\n"
            "This is the first paragraph.\n\n"
            "Second paragraph."
        )
        paragraphs = split_into_paragraphs(source_text)
        chapter_titles = extract_chapter_titles_from_raw_text(source_text)
        batches = build_batches(
            paragraphs,
            max_chars_per_batch=1000,
            chapter_titles=chapter_titles,
        )
        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0].starts_chapter)
        self.assertEqual(batches[0].chapter_title, "Prologue")

        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            runtime_options = {
                "reference_audio": "ref.wav",
                "output_wav": "out.mp3",
                "max_chars_per_batch": 1000,
                "pause_ms": 0,
                "chapter_pause_ms": 0,
                "mp3_quality": 2,
                "inference_batch_size": 1,
                "max_inference_chars": 1000,
                "max_new_tokens": 1000,
                "language": "auto",
                "tts_backend": "qwen",
                "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "device": "cpu",
                "dtype": "float16",
                "attn_implementation": "eager",
                "use_chapters": True,
                "continuation_chain": False,
                "x_vector_only_mode": False,
                "no_defrag_ui": False,
            }
            assets = create_continue_assets(
                run_dir=run_dir,
                state_path=run_dir / "state.json",
                script_path=Path(__file__).resolve().parents[1] / "audiobook_qwen3.py",
                remaining_batches=batches,
                next_batch_number=1,
                runtime_options=runtime_options,
            )
            continue_text = Path(assets["remaining_text_file"]).read_text(encoding="utf-8")

        # Re-parsing the generated continue file should preserve paragraph semantics
        # rather than introducing a duplicate title paragraph after the chapter tag.
        self.assertEqual(split_into_paragraphs(continue_text), paragraphs)


if __name__ == "__main__":
    unittest.main()
