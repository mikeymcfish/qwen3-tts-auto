import tempfile
import unittest
from pathlib import Path

from audiobook_qwen3 import find_matching_reference_text_file


class ReferenceTextLookupTests(unittest.TestCase):
    def test_returns_matching_txt_for_local_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "voice_ref.wav"
            text_path = tmp_path / "voice_ref.txt"
            audio_path.write_bytes(b"audio")
            text_path.write_text("hello", encoding="utf-8")

            found = find_matching_reference_text_file(str(audio_path))
            self.assertEqual(found, text_path.resolve())

    def test_returns_none_when_matching_txt_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "voice_ref.wav"
            audio_path.write_bytes(b"audio")

            found = find_matching_reference_text_file(str(audio_path))
            self.assertIsNone(found)

    def test_returns_none_for_non_local_reference_string(self) -> None:
        found = find_matching_reference_text_file("https://example.com/voice_ref.wav")
        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
