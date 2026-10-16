"""CPU packaging and real-shell integration checks; no Docker or GPU calls."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def shell(self, body, env=None, check=True):
        clean = {k: v for k, v in os.environ.items() if not k.startswith(('FP8KV', 'STEP5_', 'KV_DTYPE'))}
        clean.update(env or {})
        return subprocess.run(['bash', '-euo', 'pipefail', '-c', body], cwd=ROOT,
                              env=clean, capture_output=True, text=True, check=check)

    def args(self, env):
        p = self.shell('source integration/launch.fragment.sh; '
                       '''python3 -c 'import sys,json; print(json.dumps(sys.argv[1:]))' "${FP8KV_ARGS[@]}" "${FP8KV_VLLM_ARGS[@]}"''', env)
        return json.loads(p.stdout.splitlines()[-1])

    def enabled(self, **extra):
        return dict(STEP5_FP8KV='1', KV_DTYPE='fp8_e4m3', FP8KV_BLOCKS='30000',
                    FP8KV_DIR=str(ROOT / 'overlay'), **extra)

    def test_manifest_covers_exact_six_sources(self):
        seen = set()
        for line in (ROOT / 'overlay.sha256').read_text().splitlines():
            digest, name = line.split('  ')
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
            seen.add(name)
        self.assertEqual(len(seen), 6)
        self.assertEqual(seen, {str(p.relative_to(ROOT)) for p in (ROOT / 'overlay').rglob('*.py')})

    def test_disabled_adds_no_args(self):
        for val in ('0', '', 'true', '2'):
            self.assertEqual(self.args({'STEP5_FP8KV': val}), [])

    def test_enabled_mount_and_cli(self):
        args = self.args(self.enabled())
        self.assertIn('type=bind,source=' + str(ROOT / 'overlay') + ',target=/opt/step5-fp8kv,readonly', args)
        self.assertIn('PYTHONPATH=/opt/step5-fp8kv', args)
        self.assertIn('STEP5_FP8KV=1', args)
        self.assertEqual(args[-4:], ['--block-size', '16', '--num-gpu-blocks-override', '30000'])

    def test_preserves_explicit_container_pythonpath(self):
        self.assertIn('PYTHONPATH=/opt/step5-fp8kv:/opt/extra', self.args(self.enabled(FP8KV_EXTRA_PYTHONPATH='/opt/extra')))

    def test_rejects_invalid_settings(self):
        for key, value in [('FP8KV_DIR', 'overlay'), ('FP8KV_BLOCKS', '0'),
                           ('FP8KV_BLOCKS', 'x'), ('KV_DTYPE', 'auto'), ('STEP5_DENSE', '0')]:
            env = self.enabled(); env[key] = value
            self.assertNotEqual(self.shell('source integration/launch.fragment.sh', env, check=False).returncode, 0)

    def test_remote_assignments_preserve_values(self):
        env = self.enabled(FP8KV_EXTRA_PYTHONPATH='/opt/a b:/opt/c')
        p = self.shell('source integration/boot.fragment.sh; '
                       '''envline=$(fp8kv_remote_env); eval "env $envline python3 -c 'import os,json; print(json.dumps(dict(os.environ)))'"''', env)
        got = json.loads(p.stdout)
        for key, value in env.items():
            self.assertEqual(got[key], value)

    def test_example_environment(self):
        p = self.shell('source env.example; '
                       '''python3 -c 'import os,json; print(json.dumps(dict(os.environ)))' ''')
        env = json.loads(p.stdout)
        for key, value in self.enabled().items():
            self.assertEqual(env[key], value)
        self.assertEqual(env['MAXLEN'], '442368')

    def test_shell_syntax(self):
        for path in [*ROOT.rglob('*.sh'), ROOT / 'env.example']:
            subprocess.run(['bash', '-n', str(path)], check=True)


if __name__ == '__main__':
    unittest.main()
