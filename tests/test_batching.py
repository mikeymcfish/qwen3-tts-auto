import unittest

from audiobook_qwen3 import (
    build_batches,
    extract_chapter_titles_from_raw_text,
    split_into_paragraphs,
)


class BatchingTests(unittest.TestCase):
    def test_split_into_paragraphs(self) -> None:
        text = "Para one line 1\nline 2\n\n\n  Para two \n\nPara three"
        self.assertEqual(
            split_into_paragraphs(text),
            ["Para one line 1 line 2", "Para two", "Para three"],
        )

    def test_split_into_paragraphs_extracts_control_tags(self) -> None:
        text = "Para one [BREAK] para two\n\n[CHAPTER]\n\nPara three"
        self.assertEqual(
            split_into_paragraphs(text),
            ["Para one", "[BREAK]", "para two", "[CHAPTER]", "Para three"],
        )

    def test_split_into_paragraphs_extracts_speaker_tags_when_enabled(self) -> None:
        text = "Intro [S1] Hello there. [S2] General Kenobi."
        self.assertEqual(
            split_into_paragraphs(text, split_speaker_tags=True),
            ["Intro", "[S1]", "Hello there.", "[S2]", "General Kenobi."],
        )

    def test_extract_chapter_titles_from_raw_text(self) -> None:
        text = (
            "Intro\n"
            "[CHAPTER] Title One\n"
            "Body\n"
            "[chapter]   Title Two  \n"
            "[CHAPTER]\n"
            "[CHAPTER] Final Title"
        )
        self.assertEqual(
            extract_chapter_titles_from_raw_text(text),
            ["Title One", "Title Two", "", "Final Title"],
        )

    def test_batches_only_break_between_paragraphs(self) -> None:
        paragraphs = ["a" * 60, "b" * 60, "c" * 60]
        batches = build_batches(paragraphs, max_chars_per_batch=130)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].start_paragraph, 0)
        self.assertEqual(batches[0].end_paragraph, 1)
        self.assertEqual(batches[1].start_paragraph, 2)
        self.assertEqual(batches[1].end_paragraph, 2)

    def test_long_paragraph_splits_on_sentence_boundaries(self) -> None:
        sentence_a = "A" * 60 + "."
        sentence_b = "B" * 60 + "."
        sentence_c = "C" * 60 + "."
        paragraphs = [f"{sentence_a} {sentence_b} {sentence_c}"]
        batches = build_batches(paragraphs, max_chars_per_batch=130)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].text, f"{sentence_a} {sentence_b}")
        self.assertEqual(batches[1].text, sentence_c)
        self.assertEqual(batches[0].start_paragraph, 0)
        self.assertEqual(batches[0].end_paragraph, 0)
        self.assertEqual(batches[1].start_paragraph, 0)
        self.assertEqual(batches[1].end_paragraph, 0)

    def test_long_sentence_falls_back_to_hard_split(self) -> None:
        paragraphs = ["x" * 250]
        batches = build_batches(paragraphs, max_chars_per_batch=120)
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(batch.char_count <= 120 for batch in batches))
        self.assertEqual("".join(batch.text for batch in batches), "x" * 250)

    def test_break_tag_forces_batch_boundary(self) -> None:
        paragraphs = ["Hello there.", "[BREAK]", "General Kenobi."]
        batches = build_batches(paragraphs, max_chars_per_batch=200)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].text, "Hello there.")
        self.assertEqual(batches[1].text, "General Kenobi.")
        self.assertFalse(batches[0].forced_break_before)
        self.assertTrue(batches[1].forced_break_before)
        self.assertFalse(batches[1].starts_chapter)

    def test_chapter_tag_forces_break_and_marks_next_batch(self) -> None:
        paragraphs = ["First block.", "[CHAPTER]", "Second block."]
        batches = build_batches(
            paragraphs,
            max_chars_per_batch=200,
            chapter_titles=["Second block."],
        )
        self.assertEqual(len(batches), 2)
        self.assertFalse(batches[0].starts_chapter)
        self.assertTrue(batches[1].starts_chapter)
        self.assertTrue(batches[1].forced_break_before)
        self.assertEqual(batches[1].chapter_title, "Second block.")

    def test_speaker_tags_assign_batch_speakers_and_force_turn_boundaries(self) -> None:
        paragraphs = split_into_paragraphs(
            "[S1] Hello there.\n\n[S2]\nGeneral Kenobi.\n\n[S1] You are a bold one.",
            split_speaker_tags=True,
        )
        batches = build_batches(
            paragraphs,
            max_chars_per_batch=500,
            enable_speaker_tags=True,
        )
        self.assertEqual([batch.speaker_id for batch in batches], [1, 2, 1])
        self.assertEqual(
            [batch.text for batch in batches],
            ["Hello there.", "General Kenobi.", "You are a bold one."],
        )


if __name__ == "__main__":
    unittest.main()
