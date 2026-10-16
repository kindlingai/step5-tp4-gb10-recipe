"""CPU contracts only. Stubbed imports are not GPU compatibility evidence."""
from pathlib import Path
import runpy
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class BackendContractTests(unittest.TestCase):
    def load(self, sm121=True):
        platform = ModuleType("vllm.platforms")
        platform.current_platform = SimpleNamespace(is_device_capability=lambda x: sm121 and x == 121)
        source = ModuleType("step5_fp8kv.staged_backend")

        class Stock:
            @classmethod
            def get_supported_head_sizes(cls):
                return [192]
            @staticmethod
            def get_name():
                return "TRITON_ATTN"
            @staticmethod
            def get_impl_cls():
                return "stock-implementation"

        source.Step5StagedBackend = Stock
        with patch.dict(sys.modules, {platform.__name__: platform, source.__name__: source}):
            return runpy.run_path(str(ROOT / "overlay/step5_fp8kv/backend.py")), Stock

    def test_selection_uses_staged_backend(self):
        ns, stock = self.load()
        cls = ns["Step5StagedBackend"]
        self.assertEqual(cls.get_supported_head_sizes(), [192])
        self.assertEqual(stock.get_supported_head_sizes(), [192])
        self.assertEqual(cls.get_name(), "TRITON_ATTN")
        self.assertEqual(cls.get_impl_cls(), "stock-implementation")

    def test_target_cache_dtypes(self):
        ns, stock = self.load()
        for dtype in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            self.assertIs(ns["select_backend"](stock, 192, SimpleNamespace(cache_dtype=dtype)),
                          ns["Step5StagedBackend"])
        for dtype in ("nvfp4", "fp8_e5m2", "float16"):
            with self.assertRaises(ValueError):
                ns["select_backend"](stock, 192, SimpleNamespace(cache_dtype=dtype))

    def test_unsupported_head_fails(self):
        ns, stock = self.load()
        with self.assertRaises(ValueError):
            ns["select_backend"](stock, 128, None)

    def test_other_gpu_retains_stock(self):
        ns, stock = self.load(sm121=False)
        self.assertIs(ns["select_backend"](stock, 128, None), stock)
