import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))
from step5_fp8kv import hook


class HookTests(unittest.TestCase):
    def child(self, value, code):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "overlay"),
                   PYTHONDONTWRITEBYTECODE="1")
        env.pop("STEP5_FP8KV", None)
        if value is not None:
            env["STEP5_FP8KV"] = value
        return subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True, check=True).stdout

    def test_default_and_non_one_values_are_off(self):
        for value in (None, "0", "true", "2", ""):
            result = self.child(value, "import sys; print(any(type(x).__name__ == 'Step5FP8KVFinder' for x in sys.meta_path))")
            self.assertEqual(result.strip(), "False")

    def test_opt_in_does_not_import_torch_or_vllm(self):
        result = self.child("1", "import sys; print(sum(type(x).__name__ == 'Step5FP8KVFinder' for x in sys.meta_path)); print('torch' in sys.modules, 'vllm' in sys.modules)")
        self.assertEqual(result.strip(), "1\nFalse False")

    def test_install_is_idempotent(self):
        result = self.child("1", "import sys; from step5_fp8kv.hook import install; install(); install(); print(sum(type(x).__name__ == 'Step5FP8KVFinder' for x in sys.meta_path))")
        self.assertEqual(result.strip(), "1")

    def test_only_step5_model_is_intercepted(self):
        finder = hook.Step5FP8KVFinder()
        for name in ("torch", "vllm", "vllm.models.step5.nvidia.sparse_attention", "other.model"):
            self.assertIsNone(finder.find_spec(name))

    def test_source_hash_fails_closed(self):
        with self.assertRaisesRegex(ImportError, "hash mismatch"):
            hook.patch_source(b"arbitrary source")

    def test_one_backend_rewrite_preserves_dsa_branch(self):
        source = ("class Model:\n    def init(self):\n        if self.use_dsa_backend:\n            attn_backend = Step5DSAAttentionBackend\n        else:\n" + hook.OLD).encode()
        with patch.object(hook, "SOURCE_SHA256", hashlib.sha256(source).hexdigest()):
            result = hook.patch_source(source)
        ast.parse(result)
        self.assertIn("attn_backend = Step5DSAAttentionBackend", result)
        self.assertEqual(result.count("select_backend("), 1)

    def test_duplicate_and_missing_anchor_rejected(self):
        for source in (b"pass\n", (hook.OLD * 2).encode()):
            with patch.object(hook, "SOURCE_SHA256", hashlib.sha256(source).hexdigest()):
                with self.assertRaisesRegex(ImportError, "exactly one"):
                    hook.patch_source(source)

    def test_source_loader_ignores_original_pyc(self):
        source = ("if True:\n    if True:\n        if True:\n" + hook.OLD).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.py"
            path.write_bytes(source)
            loader = hook.PatchedLoader("fixture", str(path))
            with patch.object(hook, "SOURCE_SHA256", hashlib.sha256(source).hexdigest()):
                code = loader.get_code("fixture")
            self.assertIn("step5_fp8kv.backend", code.co_names)
            self.assertFalse((Path(tmp) / "__pycache__").exists())

    def test_all_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
