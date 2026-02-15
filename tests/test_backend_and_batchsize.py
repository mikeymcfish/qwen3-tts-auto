import unittest

from audiobook_qwen3 import (
    apply_continuation_chain_constraints,
    choose_inference_batch_size,
    is_cuda_oom_error,
    resolve_tts_backend,
)


class _FakeCudaProps:
    def __init__(self, total_memory_bytes: int) -> None:
        self.total_memory = total_memory_bytes


class _FakeCuda:
    def __init__(self, available: bool, total_memory_gb: float) -> None:
        self._available = available
        self._props = _FakeCudaProps(int(total_memory_gb * (1024**3)))

    def is_available(self) -> bool:
        return self._available

    def get_device_properties(self, _index: int) -> _FakeCudaProps:
        return self._props


class _FakeTorch:
    def __init__(self, cuda_available: bool, total_memory_gb: float) -> None:
        self.cuda = _FakeCuda(cuda_available, total_memory_gb)


class BackendAndBatchSizeTests(unittest.TestCase):
    def test_resolve_backend_auto_uses_model_id(self) -> None:
        self.assertEqual(
            resolve_tts_backend("auto", "OpenMOSS-Team/MOSS-TTS"),
            "moss-delay",
        )
        self.assertEqual(
            resolve_tts_backend("auto", "OpenMOSS-Team/MOSS-TTS-Local-Transformer"),
            "moss-local",
        )
        self.assertEqual(
            resolve_tts_backend("auto", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            "qwen",
        )

    def test_qwen_forces_inference_batch_size_to_one(self) -> None:
        size, warning = choose_inference_batch_size(
            requested=6,
            backend="qwen",
            torch=_FakeTorch(cuda_available=True, total_memory_gb=24.0),
            device="cuda:0",
            max_chars_per_batch=1800,
        )
        self.assertEqual(size, 1)
        self.assertIsNotNone(warning)

    def test_moss_auto_batch_size_uses_gpu_memory_and_char_budget(self) -> None:
        size, warning = choose_inference_batch_size(
            requested=0,
            backend="moss-delay",
            torch=_FakeTorch(cuda_available=True, total_memory_gb=24.0),
            device="cuda:0",
            max_chars_per_batch=1000,
        )
        self.assertEqual(size, 3)
        self.assertIsNotNone(warning)
        self.assertIn("Auto inference batch size selected 3", warning or "")

    def test_moss_auto_batch_size_on_cpu_is_one(self) -> None:
        size, warning = choose_inference_batch_size(
            requested=0,
            backend="moss-delay",
            torch=_FakeTorch(cuda_available=False, total_memory_gb=0.0),
            device="cpu",
            max_chars_per_batch=1800,
        )
        self.assertEqual(size, 1)
        self.assertIsNotNone(warning)

    def test_cuda_oom_detector(self) -> None:
        self.assertTrue(is_cuda_oom_error(RuntimeError("CUDA out of memory.")))
        self.assertTrue(is_cuda_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")))
        self.assertFalse(is_cuda_oom_error(RuntimeError("network timeout")))

    def test_continuation_chain_forces_single_batch(self) -> None:
        size, warning = apply_continuation_chain_constraints(
            backend="moss-delay",
            continuation_chain=True,
            inference_batch_size=4,
        )
        self.assertEqual(size, 1)
        self.assertIsNotNone(warning)

    def test_continuation_chain_rejects_qwen_backend(self) -> None:
        with self.assertRaises(ValueError):
            apply_continuation_chain_constraints(
                backend="qwen",
                continuation_chain=True,
                inference_batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
