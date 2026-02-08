import unittest
from unittest import mock

from audiobook_qwen3 import choose_attention_implementation


class AttentionSelectionTests(unittest.TestCase):
    def test_keeps_non_flash_backend(self) -> None:
        attn, warning = choose_attention_implementation("sdpa")
        self.assertEqual(attn, "sdpa")
        self.assertIsNone(warning)

    def test_falls_back_when_flash_attn_missing(self) -> None:
        with mock.patch(
            "audiobook_qwen3.importlib.util.find_spec", return_value=None
        ):
            attn, warning = choose_attention_implementation("flash_attention_2")
        self.assertEqual(attn, "sdpa")
        self.assertIsNotNone(warning)
        self.assertIn("not installed", warning or "")

    def test_falls_back_on_undefined_symbol_mismatch(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name in ("flash_attn", "flash_attn_2_cuda"):
                return object()
            return None

        with mock.patch(
            "audiobook_qwen3.importlib.util.find_spec",
            side_effect=fake_find_spec,
        ), mock.patch(
            "audiobook_qwen3.importlib.import_module",
            side_effect=OSError("undefined symbol: bad_abi"),
        ):
            attn, warning = choose_attention_implementation("flash_attention_2")
        self.assertEqual(attn, "sdpa")
        self.assertIsNotNone(warning)
        self.assertIn("incompatible", warning or "")

    def test_uses_flash_backend_when_probe_import_succeeds(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name in (
                "flash_attn",
                "flash_attn_2_cuda",
                "flash_attn.flash_attn_interface",
            ):
                return object()
            return None

        with mock.patch(
            "audiobook_qwen3.importlib.util.find_spec",
            side_effect=fake_find_spec,
        ), mock.patch(
            "audiobook_qwen3.importlib.import_module", return_value=object()
        ):
            attn, warning = choose_attention_implementation("flash_attention_2")
        self.assertEqual(attn, "flash_attention_2")
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
