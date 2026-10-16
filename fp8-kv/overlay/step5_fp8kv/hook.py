"""Change one explicit backend selection in the pinned Step5 model.

No source files in the image are edited. A source mismatch raises ImportError
when the model is imported, rather than silently applying an untested patch.
"""
import hashlib
import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "vllm.models.step5.nvidia.model"
SOURCE_SHA256 = "6e8574c31c421b4ebeaff36a8faa3a219cd92744251c477d43df1912a8dfa6f0"
OLD = "            attn_backend = Step5SplitKVFlashAttentionBackend\n"
NEW = """            from step5_fp8kv.backend import select_backend

            attn_backend = select_backend(
                Step5SplitKVFlashAttentionBackend, self.head_dim, cache_config
            )
"""


def patch_source(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ImportError("STEP5_FP8KV source hash mismatch; use the pinned image")
    text = source.decode("utf-8")
    if text.count(OLD) != 1:
        raise ImportError("STEP5_FP8KV expected exactly one backend selection")
    return text.replace(OLD, NEW)


class PatchedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Do not read or write a cached .pyc for a transformed source module.
        source = self.get_data(self.path)
        return compile(patch_source(source), self.path, "exec", dont_inherit=True)


class Step5FP8KVFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError("STEP5_FP8KV requires a filesystem Step5 source module")
        spec.loader = PatchedLoader(fullname, spec.origin)
        return spec


def install():
    if os.environ.get("STEP5_FP8KV") != "1":
        return
    if not any(isinstance(f, Step5FP8KVFinder) for f in sys.meta_path):
        if TARGET in sys.modules:
            raise RuntimeError("STEP5_FP8KV must be enabled before the model import")
        sys.meta_path.insert(0, Step5FP8KVFinder())
