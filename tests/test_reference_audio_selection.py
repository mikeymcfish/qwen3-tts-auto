import io
import tempfile
import unittest
from pathlib import Path

from audiobook_qwen3 import (
    prompt_for_reference_audio_selection,
    scan_reference_audio_candidates,
)


class ReferenceAudioSelectionTests(unittest.TestCase):
    def test_scan_reference_audio_candidates_returns_only_audio_with_matching_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "voice1.wav").write_bytes(b"a")
            (root / "voice1.txt").write_text("t1", encoding="utf-8")
            (root / "voice2.mp3").write_bytes(b"a")
            (root / "voice2.txt").write_text("t2", encoding="utf-8")
            (root / "voice3.flac").write_bytes(b"a")
            (root / "not_audio.txt").write_text("x", encoding="utf-8")
            (root / "voice4.ogg").write_bytes(b"a")
            (root / "voice4.md").write_text("x", encoding="utf-8")

            candidates = scan_reference_audio_candidates(root)
            names = [audio.name for audio, _ in candidates]
            self.assertEqual(names, ["voice1.wav", "voice2.mp3"])

    def test_prompt_for_reference_audio_selection_accepts_valid_choice(self) -> None:
        candidates = [
            (Path("/tmp/a.wav"), Path("/tmp/a.txt")),
            (Path("/tmp/b.mp3"), Path("/tmp/b.txt")),
        ]
        output = io.StringIO()
        selected = prompt_for_reference_audio_selection(
            candidates=candidates,
            scan_dir=Path("/tmp"),
            input_fn=lambda _prompt: "2",
            output_stream=output,
        )
        self.assertEqual(selected, Path("/tmp/b.mp3"))

    def test_prompt_for_reference_audio_selection_retries_and_can_cancel(self) -> None:
        candidates = [
            (Path("/tmp/a.wav"), Path("/tmp/a.txt")),
        ]
        answers = iter(["9", "q"])
        output = io.StringIO()
        selected = prompt_for_reference_audio_selection(
            candidates=candidates,
            scan_dir=Path("/tmp"),
            input_fn=lambda _prompt: next(answers),
            output_stream=output,
        )
        self.assertIsNone(selected)
        self.assertIn("Invalid selection", output.getvalue())


if __name__ == "__main__":
    unittest.main()
