"""Standard-library-only mode tests with a local SSE server and fake clock reader."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
BENCH_PATH=HERE/'bench.py' if (HERE/'bench.py').exists() else HERE.parent/'scripts/bench.py'
spec=importlib.util.spec_from_file_location('mode_bench',BENCH_PATH)
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

class Stub(BaseHTTPRequestHandler):
    active=maximum=completed=0
    lock=threading.Lock()
    fail_http=False
    reasoning=True
    answer='Two plus two equals four.'
    counts={}
    def log_message(self,*args):pass
    def do_GET(self):
        self.send_response(200);self.end_headers()
        if self.path=='/v1/models':self.wfile.write(b'{"data":[{"id":"step5"}]}')
        else:
            n=type(self).completed
            self.wfile.write(('\n'.join(f'vllm:spec_decode_num_{k}_total {v*n}' for k,v in [('draft_tokens',30),('accepted_tokens',20),('drafts',10)])+'\n').encode())
    def do_POST(self):
        p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        cls=type(self)
        with cls.lock:cls.active+=1;cls.maximum=max(cls.maximum,cls.active)
        try:
            time.sleep(.015)
            if cls.fail_http:
                self.send_response(503);self.end_headers();self.wfile.write(b'unavailable');return
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            def send(value):
                self.wfile.write(('data: '+json.dumps(value)+'\n\n').encode());self.wfile.flush()
            send({'choices':[{'delta':{'role':'assistant'}}]})
            if cls.reasoning:send({'choices':[{'delta':{'reasoning_content':'Compute two plus two.'}}]})
            time.sleep(.015)
            smoke=p['messages'][0]['content']=='smoke'
            send({'choices':[{'delta':{'content':cls.answer if smoke else 'A clear answer.'}}]})
            send({'choices':[{'delta':{},'finish_reason':'stop' if smoke else 'length'}]})
            count=16 if smoke else p['max_tokens']
            send({'choices':[],'usage':{'prompt_tokens':cls.counts[p['messages'][0]['content']],'completion_tokens':count}})
            with cls.lock:cls.completed+=1
            self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
        finally:
            with cls.lock:cls.active-=1

class ModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Stub)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}/v1'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)
        def case(name,size=5,output=512):
            Stub.counts[name]=size
            return {'phase':name,'rep':1,'warmup':False,'messages':[{'role':'user','content':name}],
                    'max_tokens':output,'prompt_tokens_local':size}
        smoke=case('smoke');smoke['allow_early_stop']=True
        short=case('short',output=64);short['allow_early_stop']=True
        self.fixture={'model':'step5','tokenizer_revision':'test','tokenizer_sha256':{},'cases':[],
                      's4_batches':[[case(f'b{r}s{s}') for s in range(4)] for r in range(3)],
                      'smoke_case':smoke,'soak_cases':[short,case('prefill8k',8192,1),case('decode')]}
        (self.folder/'prompts.json').write_text(json.dumps(self.fixture))
        Stub.maximum=0
    def run_mode(self,mode,duration=.25):
        args=argparse.Namespace(mode=mode,base=self.base,prompts=str(self.folder/'prompts.json'),
                                output=str(self.folder/'receipt.json'),metrics_settle=0,soak_seconds=duration,request_timeout=2)
        clock=lambda:{k:{'verified':True,'cap_state':'inactive','stdout':'STUB'} for k in bench.NODES}
        result=bench.run(args,clock_reader=clock)
        self.assertTrue((self.folder/'receipt.md').exists())
        self.assertEqual(json.loads((self.folder/'receipt.json').read_text())['passed'],result['passed'])
        return result
    def test_s4_concurrent_and_aggregate(self):
        r=self.run_mode('s4');self.assertTrue(r['passed']);self.assertEqual(Stub.maximum,4)
        runs=r['phases'][0]['runs'];self.assertEqual(len(runs),3)
        for entry in runs:
            b=entry['batch'];self.assertEqual(len(b['streams']),4)
            self.assertEqual(len(b['per_stream_tok_s']),4)
            self.assertAlmostEqual(b['aggregate_decode_tok_s'],4*511/b['aggregate_decode_span_s'])
            self.assertAlmostEqual(entry['mtp']['acceptance_fraction'],2/3)
    def test_s4_http_failure(self):
        with patch.object(Stub,'fail_http',True):r=self.run_mode('s4')
        self.assertFalse(r['passed'])
        b=r['phases'][0]['runs'][0]['batch']
        self.assertIsNone(b['aggregate_decode_tok_s']);self.assertEqual(b['streams'][0]['error']['http_status'],503)
    def test_smoke_pass(self):
        r=self.run_mode('smoke');self.assertTrue(r['passed']);self.assertTrue(all(r['smoke']['checks'].values()))
    def test_smoke_incoherent_fails(self):
        with patch.object(Stub,'answer','Bananas are an unknown integer.'):r=self.run_mode('smoke')
        self.assertFalse(r['passed']);self.assertFalse(r['smoke']['checks']['coherent_expected_answer'])
    def test_smoke_missing_reasoning_fails(self):
        with patch.object(Stub,'reasoning',False):r=self.run_mode('smoke')
        self.assertFalse(r['passed']);self.assertFalse(r['smoke']['checks']['reasoning_field_present'])
    def test_soak_coverage_and_log(self):
        r=self.run_mode('soak');self.assertTrue(r['passed'])
        s=r['soak'];self.assertGreaterEqual(s['duration_s'],.25)
        self.assertEqual(s['concurrency_seen'],[1,2,3,4]);self.assertEqual(len(s['workloads_seen']),3)
        records=[json.loads(line) for line in (self.folder/'receipt.requests.jsonl').read_text().splitlines()]
        self.assertEqual(len(records),s['request_count']);self.assertTrue(all(x['passed'] for x in records))
    def test_soak_http_errors_fail_and_log(self):
        with patch.object(Stub,'fail_http',True):r=self.run_mode('soak')
        self.assertFalse(r['passed']);self.assertGreater(len(r['soak']['errors']),0)
        records=[json.loads(line) for line in (self.folder/'receipt.requests.jsonl').read_text().splitlines()]
        self.assertTrue(all(x['error']['http_status']==503 for x in records))
    def test_soak_incomplete_coverage_fails(self):
        r=self.run_mode('soak',duration=.001)
        self.assertFalse(r['passed']);self.assertFalse(r['soak']['coverage_complete'])

if __name__=='__main__':unittest.main(verbosity=2)
