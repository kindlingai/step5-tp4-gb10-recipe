#!/usr/bin/env python3
"""Standard-library-only S1 client. No GPU work or container changes on client host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import os
import statistics
import subprocess
import time
import urllib.request
import uuid

CLOCK_CMD = 'nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv; gpu_rc=$?; systemctl is-active gpu-clock-cap; cap_rc=$?; printf "GPU_RC=%s CAP_RC=%s\\n" "$gpu_rc" "$cap_rc"'
NODES = {f'rank{i}':host for i,host in enumerate(os.environ.get('BENCH_HOSTS', '192.168.50.11,192.168.50.12,192.168.50.13,192.168.50.14').split(','))}
SSH_USER = os.environ.get('SSH_USER', '')

def utc(): return datetime.now(timezone.utc).isoformat()

def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()

def metrics(url):
    raw = get(url)
    values = {}
    for line in raw.splitlines():
        m = re.match(r'([^\s{]+)(\{.*?\})?\s+([-+\d.eE]+)(?:\s|$)',line)
        if m and ('spec_decode' in m[1] or any(s in m[1] for s in ('requests_running','requests_waiting','request_success_total','prefix_cache'))):
            value = float(m[3])
            if math.isfinite(value): values[m[1]+(m[2] or '')] = value
    return {'utc':utc(),'values':values,'raw':raw}

def mtp_delta(before, after):
    b,a=before['values'],after['values']
    # A new series starts at zero; disappeared or decreasing counters invalidate the ratio.
    delta={k:a[k]-b.get(k,0) for k in a if 'spec_decode' in k and '_created' not in k}
    reset=any(v<0 for v in delta.values()) or any(k not in a for k in b if 'spec_decode' in k)
    def total(name):
        vals=[v for k,v in delta.items() if k.split('{')[0] in (name,name+'_total')]
        return sum(vals) if vals else None
    draft=total('vllm:spec_decode_num_draft_tokens')
    accepted=total('vllm:spec_decode_num_accepted_tokens')
    steps=total('vllm:spec_decode_num_drafts')
    return {'counter_delta':delta, 'counter_reset_or_disappearance':reset,
            'draft_tokens':draft,'accepted_tokens':accepted,'draft_steps':steps,
            'acceptance_fraction':accepted/draft if not reset and draft and accepted is not None else None,
            'mean_acceptance_length':1+accepted/steps if not reset and steps and accepted is not None else None,
            'attribution':'Global server counters; per-run only if no other traffic.'}

def clocks():
    def one(item):
        name,host=item
        start=utc()
        try:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',f'{SSH_USER}@{host}' if SSH_USER else host,CLOCK_CMD],capture_output=True,text=True,timeout=25)
            lines=p.stdout.splitlines()
            cap=next((s for s in lines if s in ('active','inactive','failed','activating','deactivating','unknown')),None)
            gpu_ok='GPU_RC=0 ' in p.stdout
            return name,{'utc':start,'host':host,'ssh_rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr,
                         'cap_state':cap,'verified':p.returncode==0 and gpu_ok and cap is not None and ('CAP_RC=0' in p.stdout or 'CAP_RC=3' in p.stdout)}
        except Exception as e: return name,{'utc':start,'host':host,'verified':False,'error':repr(e)}
    with ThreadPoolExecutor(max_workers=4) as ex: return dict(ex.map(one,NODES.items()))

def clock_label(before,after):
    snapshots=[s for snap in (before,after) for s in snap.values()]
    if len(snapshots)!=8 or not all(s.get('verified') for s in snapshots): return 'unknown'
    if all(s['cap_state']=='inactive' for s in snapshots): return 'fullclk'
    return 'cap-active-or-mixed'

def stream(base, payload, expected_prompt_tokens):
    request=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start_utc=utc(); t0=time.perf_counter(); events=[]; first=None; last=None; usage=None; finish=None; done=False
    texts={'content':[], 'reasoning_content':[], 'reasoning':[]}
    with urllib.request.urlopen(request,timeout=900) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            data=line[5:].strip()
            if data==b'[DONE]': done=True; break
            event=json.loads(data)
            elapsed=time.perf_counter()-t0
            events.append({'t_s':elapsed,'event':event})
            if event.get('error'): raise RuntimeError(event['error'])
            if event.get('usage'): usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta') or {}
                emitted=False
                for key in texts:
                    if delta.get(key): texts[key].append(delta[key]); emitted=True
                if emitted:
                    if first is None: first=elapsed
                    last=elapsed
                if choice.get('finish_reason'): finish=choice['finish_reason']
    duration=time.perf_counter()-t0
    if not done: raise RuntimeError('Stream ended without [DONE]')
    if usage is None: raise RuntimeError('Missing usage; generated-token count cannot be verified')
    if first is None: raise RuntimeError('No content/reasoning text in stream; TTFT cannot be measured')
    n=usage['completion_tokens']; dt=last-first
    warnings=[]
    if usage['prompt_tokens']!=expected_prompt_tokens: warnings.append('SERVER_PROMPT_TOKEN_COUNT_MISMATCH')
    if n!=payload['max_tokens']: warnings.append('OUTPUT_LENGTH_NOT_REQUESTED_LENGTH')
    return {'start_utc':start_utc,'end_utc':utc(),'ttft_s':first,'last_token_s':last,'total_s':duration,
            'usage':usage,'finish_reason':finish,'prompt_tok_s_proxy':usage['prompt_tokens']/first,
            'decode_first_to_last_s':dt,'decode_tok_s':(n-1)/dt if dt>0 and n>1 else None,
            'decode_formula':'(usage.completion_tokens - 1) / (last_text_SSE_time - first_text_SSE_time)',
            'decode_caveat':'SSE boundary estimate; chunks may contain multiple tokens and hidden special tokens lack text timestamps.',
            'warnings':warnings,'text':{k:''.join(v) for k,v in texts.items()},'events':events}

def stats(values):
    values=[v for v in values if v is not None]
    return {'n':len(values),'median':statistics.median(values),'mean':statistics.mean(values),'min':min(values),'max':max(values)} if values else None

def markdown(receipt):
    lines=['# Step-5 TP4 first numbers','', '| Phase | Clock | n | TTFT median (s) | Prompt tok/s proxy median | S1 decode tok/s median | MTP accepted/drafted |', '|---|---|---:|---:|---:|---:|---:|']
    for phase in receipt['phases']:
        runs=[r for r in phase['runs'] if not r['case']['warmup'] and 'result' in r]
        if not runs: continue
        def median(key):
            s=stats([r['result'][key] for r in runs]); return f"{s['median']:.2f}" if s else '—'
        pairs=[r['mtp'] for r in runs if r.get('mtp',{}).get('acceptance_fraction') is not None]
        draft=sum(p['draft_tokens'] for p in pairs); accepted=sum(p['accepted_tokens'] for p in pairs)
        rate=f'{100*accepted/draft:.1f}%' if draft else 'N/A'
        lines.append(f"| {phase['name']} | {phase['clock_label']} | {len(runs)} | {median('ttft_s')} | {median('prompt_tok_s_proxy') if phase['name'].startswith('prefill') else '—'} | {median('decode_tok_s') if phase['name'].startswith('decode') else '—'} | {rate} |")
    lines += ['', 'Prefill uses 1 warmup + 3 measured requests at each size. Decode uses 3 requests per type.',
              'Prompt tok/s is prompt_tokens / client TTFT, not a GPU-only prefill rate.',
              'Decode counts all usage completion tokens, including reasoning. SSE times are approximate token boundaries.',
              'MTP ratios use global server counter deltas. Other traffic invalidates per-request attribution.',
              'fullclk means gpu-clock-cap was inactive on all four ranks before and after the phase; raw clocks are in JSON.',
              'These are preliminary three-repeat measurements, not a quality evaluation.']
    warns=[w for p in receipt['phases'] for r in p['runs'] for w in r.get('result',{}).get('warnings',[])]
    if warns: lines += ['','Warnings: '+', '.join(sorted(set(warns)))+'.']
    if receipt.get('error'): lines += ['','INCOMPLETE: '+receipt['error']]
    return '\n'.join(lines)+'\n'

def run(args, clock_reader=clocks):
    fixture=json.loads(Path(args.prompts).read_text())
    base=args.base.rstrip('/'); metrics_url=base.rsplit('/v1',1)[0]+'/metrics'
    receipt={'schema':1,'started_utc':utc(),'client_host':os.environ.get('CLIENT_LABEL', 'client'),'client_python':__import__('sys').version,
             'endpoint':base,'model':fixture['model'],'tokenizer_revision':fixture['tokenizer_revision'],
             'tokenizer_sha256':fixture['tokenizer_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'fixture_sha256':hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
             'fixture':fixture,'mtp_config_expected':3,'metrics_settle_s':args.metrics_settle,'phases':[]}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=output.with_suffix('.tmp'); temp.write_text(json.dumps(receipt,indent=2)+'\n'); temp.replace(output)
        output.with_suffix('.md').write_text(markdown(receipt))
    try:
        receipt['models']=json.loads(get(base+'/models'))
        assert fixture['model'] in [x['id'] for x in receipt['models']['data']], 'Expected model missing'
        for name in dict.fromkeys(c['phase'] for c in fixture['cases']):
            phase={'name':name,'clock_before':clock_reader(),'clock_label':'pending','runs':[]}
            receipt['phases'].append(phase); save()
            try:
                for case in [c for c in fixture['cases'] if c['phase']==name]:
                    entry={'case':case}; phase['runs'].append(entry)
                    entry['metrics_before']=metrics(metrics_url)
                    payload={'model':fixture['model'],'messages':case['messages'],'max_tokens':case['max_tokens'],
                             'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
                    # Force a 512-token decode sample even if the model would otherwise stop early.
                    if case['max_tokens']==512: payload['ignore_eos']=True
                    entry['payload']=payload
                    entry['result']=stream(base,payload,case['prompt_tokens_local']); save()
                    # vLLM's metrics logger can publish counters asynchronously. This is outside timed requests.
                    if args.metrics_settle: time.sleep(args.metrics_settle)
                    entry['metrics_after']=metrics(metrics_url)
                    entry['mtp']=mtp_delta(entry['metrics_before'],entry['metrics_after'])
                    print(name,case['rep'],json.dumps({k:entry['result'][k] for k in ('ttft_s','decode_tok_s','usage','warnings')}),flush=True)
                    save()
            finally:
                phase['clock_after']=clock_reader()
                phase['clock_label']=clock_label(phase['clock_before'],phase['clock_after'])
                phase['summary']={k:stats([r['result'][k] for r in phase['runs'] if 'result' in r and not r['case']['warmup']]) for k in ('ttft_s','prompt_tok_s_proxy','decode_tok_s')}
                save()
        receipt['complete']=True
    except Exception as e:
        receipt['complete']=False; receipt['error']=repr(e); raise
    finally:
        receipt['ended_utc']=utc(); save()
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',default=os.environ.get('BENCH_BASE', 'http://192.168.50.11:18000/v1'))
    p.add_argument('--prompts',default=str(Path(__file__).with_name('prompts.json')))
    p.add_argument('--output',default='receipt.json')
    p.add_argument('--metrics-settle',type=float,default=11)
    p.add_argument('--go',action='store_true',help='Required: explicitly authorize benchmark requests')
    a=p.parse_args()
    if not a.go: p.error('Pass --go only when the server is ready for benchmark requests')
    run(a)
