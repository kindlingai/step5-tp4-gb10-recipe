#!/usr/bin/env python3
"""CPU-only tests. External deployment commands are never executed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]

def shell(name,*args,env=None):
    return subprocess.run(['bash',str(REPO/'scripts'/name),*map(str,args)],env=env,capture_output=True,text=True)

class RecipeTests(unittest.TestCase):
    def test_lists(self):
        groups=[(REPO/f'lists/files-holder{i}.txt').read_text().splitlines() for i in range(3)]
        small=(REPO/'lists/files-small.txt').read_text().splitlines()
        self.assertEqual([len(g) for g in groups],[21,17,57])
        all_files=sum(groups,[])+small
        self.assertEqual(len(all_files),107);self.assertEqual(len(set(all_files)),107)
        for name in all_files:
            self.assertEqual(Path(name).name,name)
    def test_links_all_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            for rank in range(4):
                storage=base/f'rank{rank}'
                for holder in range(3):
                    source=storage/'weights' if rank==holder else storage/f'nfs/holder{holder}'
                    source.mkdir(parents=True,exist_ok=True)
                    files=(REPO/f'lists/files-holder{holder}.txt').read_text().splitlines()
                    if holder==0:files+=(REPO/'lists/files-small.txt').read_text().splitlines()
                    for name in files:(source/name).write_text('fixture')
                env={**os.environ,'STEP5_ROOT':str(storage),'STEP5_CONFIG':''}
                p=shell('mklinks.sh',rank,env=env);self.assertEqual(p.returncode,0,p.stderr)
                links=list((storage/'model').iterdir());self.assertEqual(len(links),107)
                self.assertTrue(all(x.is_symlink() and x.is_file() for x in links))
                p=shell('mklinks.sh',rank,env=env);self.assertEqual(p.returncode,0,p.stderr)
    def test_launch_mock(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); storage=base/'storage';(storage/'model').mkdir(parents=True)
            for f in ('config.json','model.safetensors.index.json'):(storage/'model'/f).write_text('{}')
            mock=base/'docker'
            mock.write_text('#!/usr/bin/env python3\nimport json,os,sys\nif sys.argv[1:3]==["container","inspect"]:sys.exit(1)\nwith open(os.environ["MOCK_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nprint("mock-container")\n')
            mock.chmod(0o755)
            env={**os.environ,'STEP5_CONFIG':'','STEP5_ROOT':str(storage),'CACHE_DIR':str(base/'cache'),'PATH':str(base)+os.pathsep+os.environ['PATH'],'MOCK_LOG':str(base/'calls'),'GMU':'0.85','BATCH_TOKENS':'4096','KV_DTYPE':'auto','STEP5_DENSE':'1','LMONLY':'1','SPEC_TOKENS':'3','DROP_PAGE_CACHE':'0','STEP5_FP8KV':'0'}
            for key in ('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS','MAXLEN','RDMA_HCA','NCCL_CROSS_NIC','NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS'):env.pop(key,None)
            for rank in range(4):
                p=shell('launch.sh',rank,env=env);self.assertEqual(p.returncode,0,p.stderr)
            calls=[json.loads(x) for x in (base/'calls').read_text().splitlines()]
            self.assertEqual(len(calls),4)
            for rank,call in enumerate(calls):
                for flag,value in [('--max-model-len','196608'),('--kv-cache-dtype','auto'),('--gpu-memory-utilization','0.85'),('--max-num-batched-tokens','4096'),('--tensor-parallel-size','4'),('--nnodes','4'),('--node-rank',str(rank))]:
                    self.assertEqual(call[call.index(flag)+1],value)
                self.assertIn('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1',call)
                self.assertEqual('--headless' in call,rank!=0)
                self.assertIn('--language-model-only',call)
                for setting in ('NCCL_IB_HCA=mlx5_0,mlx5_1','NCCL_CROSS_NIC=1','NCCL_MIN_NCHANNELS=2','NCCL_MAX_NCHANNELS=2'):
                    self.assertIn(setting,call)
                self.assertEqual(call[call.index('--spec-method')+1],'mtp')
                self.assertEqual(call[call.index('--spec-tokens')+1],'3')
                self.assertEqual(json.loads(call[call.index('--hf-overrides')+1]),{'text_config':{'sparse_config':{'enabled':False}}})
                self.assertIn(f'type=bind,source={storage},target={storage},readonly',call)
            env.update(MAXLEN='65536',RDMA_HCA='rail_a,rail_b',NCCL_CROSS_NIC='0',NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4')
            p=shell('launch.sh',0,env=env);self.assertEqual(p.returncode,0,p.stderr)
            alternate=json.loads((base/'calls').read_text().splitlines()[-1])
            self.assertEqual(alternate[alternate.index('--max-model-len')+1],'65536')
            for setting in ('NCCL_IB_HCA=rail_a,rail_b','NCCL_CROSS_NIC=0','NCCL_MIN_NCHANNELS=4','NCCL_MAX_NCHANNELS=4'):
                self.assertIn(setting,alternate)
            for setting in ('MAXLEN=589824','FP8KV_BLOCKS=40000','GMU=0.86','VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0','STEP5_FP8KV=1','KV_DTYPE=fp8_e4m3','RDMA_HCA=mlx5_0,mlx5_1','NCCL_CROSS_NIC=1','NCCL_MIN_NCHANNELS=2','NCCL_MAX_NCHANNELS=2'):
                self.assertIn(setting,(REPO/'env.example').read_text().splitlines())
            # Default and safer profiles: these are CPU command tests, not GPU fit proof.
            env.update(STEP5_FP8KV='1',KV_DTYPE='fp8_e4m3',RDMA_HCA='rail_a,rail_b',
                NCCL_CROSS_NIC='1',NCCL_MIN_NCHANNELS='2',NCCL_MAX_NCHANNELS='2')
            env.pop('FP8KV_DIR',None)
            for cap,blocks,gmu,estimate in [('442368','30000','0.85','1'),('589824','40000','0.86','0')]:
                env.update(MAXLEN=cap,FP8KV_BLOCKS=blocks,GMU=gmu,
                    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=estimate)
                for rank in range(4):
                    p=shell('launch.sh',rank,env=env);self.assertEqual(p.returncode,0,p.stderr)
                    fp8=json.loads((base/'calls').read_text().splitlines()[-1])
                    for flag,value in [('--kv-cache-dtype','fp8_e4m3'),('--block-size','16'),
                        ('--num-gpu-blocks-override',blocks),('--max-model-len',cap),
                        ('--gpu-memory-utilization',gmu),('--spec-tokens','3'),
                        ('--max-num-batched-tokens','4096'),('--max-num-seqs','8'),('--node-rank',str(rank))]:
                        self.assertEqual(fp8[fp8.index(flag)+1],value)
                    setting='VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS='+estimate
                    self.assertIn(setting,fp8)
                    self.assertEqual(fp8[fp8.index(setting)-1],'-e')
                    self.assertLess(fp8.index(setting),fp8.index('--entrypoint'))
                    self.assertIn('STEP5_FP8KV=1',fp8)
                    self.assertIn('PYTHONPATH=/opt/step5-fp8kv',fp8)
                    self.assertIn('NCCL_CROSS_NIC=1',fp8)
                    self.assertIn('type=bind,source='+str(REPO/'fp8-kv/overlay')+',target=/opt/step5-fp8kv,readonly',fp8)
            before=(base/'calls').read_text()
            for field in ('MAXLEN','FP8KV_BLOCKS','GMU'):
                missing={**env,field:''}
                p=shell('launch.sh',0,env=missing);self.assertNotEqual(p.returncode,0)
                self.assertEqual((base/'calls').read_text(),before)
            env.update(STEP5_FP8KV='0',KV_DTYPE='auto',MAXLEN='262144',GMU='0.86',
                VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS='1')
            p=shell('launch.sh',0,env=env);self.assertEqual(p.returncode,0,p.stderr)
            bf16=json.loads((base/'calls').read_text().splitlines()[-1])
            self.assertEqual(bf16[bf16.index('--max-model-len')+1],'262144')
            self.assertEqual(bf16[bf16.index('--gpu-memory-utilization')+1],'0.86')
            self.assertIn('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1',bf16)
            self.assertNotIn('--num-gpu-blocks-override',bf16)
            self.assertNotIn('STEP5_FP8KV=1',bf16)
            p=shell('launch.sh',4,env=env);self.assertNotEqual(p.returncode,0)
    def test_default_profile_from_example(self):
        keys=['MAXLEN','FP8KV_BLOCKS','GMU','VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS',
              'STEP5_FP8KV','KV_DTYPE','SPEC_TOKENS','BATCH_TOKENS','MAXSEQ','STEP5_DENSE']
        p=subprocess.run(['bash','-c','set -a; source "$1"; python3 -c "$2"','bash',
            str(REPO/'env.example'),
            'import os,json; print(json.dumps([os.environ[k] for k in '+repr(keys)+']))'],
            capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout),['589824','40000','0.86','0','1','fp8_e4m3','3','4096','8','1'])
    def test_page_cache_loop_dry_run(self):
        p=shell('drop-page-cache.sh','--dry-run')
        self.assertEqual(p.returncode,0,p.stderr)
        lines=p.stdout.splitlines();self.assertEqual(len(lines),90)
        self.assertTrue(lines[0].startswith('t=0s:'))
        self.assertTrue(lines[-1].startswith('t=890s:'))
        self.assertTrue(all('echo 1 > /proc/sys/vm/drop_caches' in line for line in lines))
    def test_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);w=base/'weights';w.mkdir()
            names=(REPO/'lists/files-holder0.txt').read_text().splitlines()
            for name in names:(w/name).write_bytes(b'fixture')
            checksum=hashlib.sha256(b'fixture').hexdigest()
            (w/'SHA256SUMS').write_text(''.join(f'{checksum}  {name}\n' for name in names))
            env={**os.environ,'STEP5_CONFIG':'','STEP5_ROOT':str(base)}
            p=shell('verify-shards.sh',0,env=env);self.assertEqual(p.returncode,0,p.stderr)
            (w/names[0]).write_text('changed')
            p=shell('verify-shards.sh',0,env=env);self.assertNotEqual(p.returncode,0)
    def test_privacy_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'scripts').mkdir()
            shutil.copyfile(REPO/'scripts/privacy_scan.sh',base/'scripts/privacy_scan.sh')
            def scan():return subprocess.run(['bash',str(base/'scripts/privacy_scan.sh')],capture_output=True,text=True)
            self.assertEqual(scan().returncode,0)
            bad=['100'+'.64.1.1','10'+'.98.0.2','d'+'gx1','spa'+'rk-abcd','/ho'+'me/example','/ro'+'ot/private',
                 'reg'+'gie','pu'+'lse','As'+'tra','Fa'+'ble','O'+'pus','S'+'ol','Ar'+'gon','hf_'+'A'*25]
            for value in bad:
                (base/'example.txt').write_text(value+'\n')
                p=scan();self.assertEqual(p.returncode,1,(value,p.stdout,p.stderr))
            (base/'example.txt').unlink()
            self.assertEqual(scan().returncode,0)
    def test_bench_configuration(self):
        spec=importlib.util.spec_from_file_location('public_bench',REPO/'scripts/bench.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(len(module.NODES),4)
        self.assertEqual(list(module.NODES),['rank0','rank1','rank2','rank3'])
        snapshot={name:{'verified':True,'cap_state':'inactive'} for name in module.NODES}
        self.assertEqual(module.clock_label(snapshot,snapshot),'fullclk')
        self.assertIsNone(module.mtp_delta({'values':{}},{'values':{}})['acceptance_fraction'])

if __name__=='__main__':unittest.main(verbosity=2)
