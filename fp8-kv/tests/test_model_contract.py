"""Dependency-free regression for Step5's model-specific implementation hooks.
This test does not execute tensors, graph capture, or GPU kernels.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModelContractTests(unittest.TestCase):
    def impl(self):
        tree = ast.parse((ROOT / 'overlay/step5_fp8kv/staged_backend.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Step5StagedImpl')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                   ('_split_kv_cache', 'supports_headwise_gate')]
        self.assertEqual({n.name for n in methods}, {'_split_kv_cache', 'supports_headwise_gate'})
        isolated = ast.Module(body=[ast.ClassDef(name='Impl', bases=[], keywords=[], body=methods,
                                               decorator_list=[])], type_ignores=[])
        ns = {}
        exec(compile(ast.fix_missing_locations(isolated), '<contract-methods>', 'exec'), ns)
        impl = ns['Impl']()
        impl.head_size = 192
        return impl

    def test_split_is_original_storage_nhd_view(self):
        calls = []
        key, value = object(), object()
        class Cache:
            def transpose(self, first, second):
                calls.append(('transpose', first, second))
                return self
            def split(self, size, dim):
                calls.append(('split', size, dim))
                return key, value
        self.assertEqual(self.impl()._split_kv_cache(Cache()), (key, value))
        self.assertEqual(calls, [('transpose', 1, 2), ('split', 192, -1)])

    def test_headwise_gate_stays_on_model_fallback(self):
        self.assertIs(self.impl().supports_headwise_gate(), False)
