"""Exercise remote-shell cleanup locally with a fake nvidia-smi; never use SSH."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

HERE=Path(__file__).resolve().parent
PATH=HERE/'bench.py' if (HERE/'bench.py').exists() else HERE.parent/'scripts/bench.py'
spec=importlib.util.spec_from_file_location('telemetry_bench',PATH)
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        fake=self.folder/'nvidia-smi'
        fake.write_text('''#!/usr/bin/env python3
import os,signal,sys,time
from pathlib import Path
folder=Path(os.environ['TEST_DIR']);name=os.environ['TEST_NODE']
def stopped(*args):
    (folder/(name+'.stopped')).write_text('stopped')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stopped)
(folder/(name+'.pid')).write_text(str(os.getpid()))
while True:
    print('N/A, 80, 45' if os.environ.get('BAD_SAMPLE') else '2100, 80, 45',flush=True)
    time.sleep(.01)
''');fake.chmod(0o755)
    def command(self,node,bad=False):
        return ['env',f'PATH={self.folder}{os.pathsep}{os.environ["PATH"]}',f'TEST_DIR={self.folder}',f'TEST_NODE={node}',
                f'BAD_SAMPLE={"yes" if bad else ""}','bash','-c',bench.SAMPLER_CMD]
    def test_four_node_start_stop_and_child_cleanup(self):
        sampler=bench.PhaseSampler(nodes={f'rank{i}':f'node{i}' for i in range(4)},command_factory=self.command,startup_timeout=3).start()
        try:
            start=time.perf_counter();time.sleep(.04);end=time.perf_counter()
        finally:data=sampler.stop()
        self.assertIn('--loop=1',bench.SAMPLER_CMD)
        self.assertTrue(data['stopped']);self.assertEqual(len(data['nodes']),4)
        for i,node in enumerate(data['nodes'].values()):
            self.assertTrue(node['cleanup_confirmed'],node)
            self.assertGreaterEqual(len(node['samples']),2)
            self.assertTrue((self.folder/f'node{i}.stopped').exists())
            pid=int((self.folder/f'node{i}.pid').read_text())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        phase={'telemetry':data,'runs':[{'result':{'start_monotonic_s':start,'total_s':end-start}}]}
        summary=bench.phase_hardware(phase)
        self.assertEqual(summary['source'],'in-phase 1s')
        self.assertEqual(summary['sm_mhz']['median'],2100)
        self.assertEqual(summary['temperature_c']['max'],45)
        self.assertEqual(sampler.stop(),data)
    def test_stdin_eof_kills_remote_child(self):
        process=subprocess.Popen(self.command('eof'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            self.assertIn('2100',process.stdout.readline())
            process.stdin.close();process.wait(timeout=3)
            self.assertTrue((self.folder/'eof.stopped').exists())
        finally:
            if process.poll() is None:process.kill();process.wait()
            process.stdout.close()
    def test_invalid_telemetry_falls_back(self):
        sampler=bench.PhaseSampler(nodes={f'rank{i}':f'bad{i}' for i in range(4)},command_factory=lambda n:self.command(n,True),startup_timeout=3).start()
        data=sampler.stop()
        self.assertTrue(all(n['errors'] for n in data['nodes'].values()))
        self.assertEqual(bench.phase_hardware({'telemetry':data})['source'],'boundary fallback')
    def test_active_windows_exclude_idle_and_warmup(self):
        nodes={f'rank{i}':{'cleanup_confirmed':True,'errors':[],'samples':[
            {'monotonic_s':t,'sm_mhz':clock,'power_w':70,'temperature_c':45}
            for t,clock in [(0,500),(2,2100),(4,500)]]} for i in range(4)}
        phase={'telemetry':{'stopped':True,'nodes':nodes},'runs':[
            {'case':{'warmup':True},'result':{'start_monotonic_s':0,'total_s':.5}},
            {'result':{'start_monotonic_s':1,'total_s':2}}]}
        summary=bench.phase_hardware(phase)
        self.assertEqual(summary['source'],'in-phase 1s');self.assertEqual(summary['sm_mhz']['min'],2100)
        self.assertEqual(summary['node_sample_counts'],{f'rank{i}':1 for i in range(4)})
        nodes['rank0']['samples']=[]
        self.assertEqual(bench.phase_hardware(phase)['source'],'boundary fallback')
    def test_spawn_failure_is_recorded(self):
        sampler=bench.PhaseSampler(nodes={f'rank{i}':f'node{i}' for i in range(4)},command_factory=lambda _:['/does-not-exist'],startup_timeout=.1).start()
        data=sampler.stop()
        self.assertTrue(data['stopped'])
        self.assertTrue(all(n['errors'] and not n['cleanup_confirmed'] for n in data['nodes'].values()))

if __name__=='__main__':unittest.main(verbosity=2)
