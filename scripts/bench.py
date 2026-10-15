#!/usr/bin/env python3
"""Standard-library-only S1/S4, smoke, and soak client. No GPU work or container changes on client host."""
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
import threading
import urllib.error
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

def stream(base, payload, expected_prompt_tokens, timeout=900, keep_events=True):
    request=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start_utc=utc(); t0=time.perf_counter(); events=[]; first=None; last=None; usage=None; finish=None; done=False; reasoning_field_seen=False
    texts={'content':[], 'reasoning_content':[], 'reasoning':[]}
    with urllib.request.urlopen(request,timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            data=line[5:].strip()
            if data==b'[DONE]': done=True; break
            event=json.loads(data)
            elapsed=time.perf_counter()-t0
            if keep_events: events.append({'t_s':elapsed,'event':event})
            if event.get('error'): raise RuntimeError(event['error'])
            if event.get('usage'): usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta') or {}
                if 'reasoning_content' in delta or 'reasoning' in delta: reasoning_field_seen=True
                emitted=False
                for key in texts:
                    if delta.get(key): texts[key].append(delta[key]); emitted=True
                if emitted:
                    if first is None: first=elapsed
                    last=elapsed
                if choice.get('finish_reason'): finish=choice['finish_reason']
    duration=time.perf_counter()-t0
    if not done: raise RuntimeError('Stream ended without [DONE]')
    if finish is None: raise RuntimeError('Missing finish_reason')
    if usage is None: raise RuntimeError('Missing usage; generated-token count cannot be verified')
    if first is None: raise RuntimeError('No content/reasoning text in stream; TTFT cannot be measured')
    n=usage['completion_tokens']; dt=last-first
    warnings=[]
    if expected_prompt_tokens is not None and usage['prompt_tokens']!=expected_prompt_tokens: warnings.append('SERVER_PROMPT_TOKEN_COUNT_MISMATCH')
    if (payload.get('ignore_eos') or payload['max_tokens']==1) and n!=payload['max_tokens']: warnings.append('OUTPUT_LENGTH_NOT_REQUESTED_LENGTH')
    return {'start_utc':start_utc,'end_utc':utc(),'start_monotonic_s':t0,'reasoning_field_seen':reasoning_field_seen,'ttft_s':first,'last_token_s':last,'total_s':duration,
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
        if phase.get('kind')=='s4': continue
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
    lines += extra_markdown(receipt)
    if receipt.get('error'): lines += ['','INCOMPLETE: '+receipt['error']]
    return '\n'.join(lines)+'\n'

def run(args, clock_reader=clocks):
    fixture=json.loads(Path(args.prompts).read_text())
    base=args.base.rstrip('/'); metrics_url=base.rsplit('/v1',1)[0]+'/metrics'
    receipt={'schema':2,'started_utc':utc(),'client_host':os.environ.get('CLIENT_LABEL', 'client'),'client_python':__import__('sys').version,
             'endpoint':base,'model':fixture['model'],'tokenizer_revision':fixture['tokenizer_revision'],
             'tokenizer_sha256':fixture['tokenizer_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'fixture_sha256':hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
             'fixture':fixture,'mode':getattr(args,'mode','s1'),'mtp_config_expected':3,'metrics_settle_s':args.metrics_settle,'phases':[]}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=output.with_suffix('.tmp'); temp.write_text(json.dumps(receipt,indent=2)+'\n'); temp.replace(output)
        output.with_suffix('.md').write_text(markdown(receipt))
    try:
        receipt['models']=json.loads(get(base+'/models'))
        assert fixture['model'] in [x['id'] for x in receipt['models']['data']], 'Expected model missing'
        mode=getattr(args,'mode','s1')
        if mode=='smoke': run_smoke(args,fixture,receipt,save,clock_reader)
        if mode=='soak': run_soak(args,fixture,receipt,save,clock_reader)
        for name in (dict.fromkeys(c['phase'] for c in fixture['cases']) if mode in ('bench','s1') else []):
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
                    if entry['result']['warnings']: receipt['passed']=False
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
        if mode in ('bench','s4'): run_s4(args,fixture,receipt,save,clock_reader)
        receipt['complete']=True
        receipt.setdefault('passed',True)
    except Exception as e:
        receipt['complete']=False; receipt['passed']=False; receipt['error']=repr(e); receipt['error_detail']=error_record(e); raise
    finally:
        receipt['ended_utc']=utc(); save()
    return receipt

def payload_for(model, case):
    payload={'model':model,'messages':case['messages'],'max_tokens':case['max_tokens'],
             'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
    if case['max_tokens']==512 and not case.get('allow_early_stop'):
        payload['ignore_eos']=True
    return payload

def error_record(error):
    result={'type':type(error).__name__,'message':str(error),'utc':utc()}
    if isinstance(error,urllib.error.HTTPError):
        result['http_status']=error.code
        result['body']=error.read(4096).decode('utf-8',errors='replace')
    return result

def concurrent_batch(base, model, cases, timeout=900, keep_events=True):
    barrier=threading.Barrier(len(cases))
    def one(index,case):
        payload=payload_for(model,case)
        entry={'stream_index':index,'case':case,'payload':payload}
        try:
            barrier.wait(timeout=15)
            entry['result']=stream(base,payload,case.get('prompt_tokens_local'),timeout,keep_events)
            entry['passed']=not entry['result']['warnings']
        except Exception as error:
            entry['passed']=False;entry['error']=error_record(error)
        if not keep_events:
            entry['case']={k:v for k,v in case.items() if k!='messages'}
            entry.pop('payload',None)
        return entry
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(cases)) as ex:
        futures=[ex.submit(one,i,c) for i,c in enumerate(cases)]
        entries=[f.result() for f in futures]
    wall=time.perf_counter()-start
    successful=[e['result'] for e in entries if 'result' in e]
    all_pass=all(e['passed'] for e in entries)
    first=min((r['start_monotonic_s']+r['ttft_s'] for r in successful),default=None)
    last=max((r['start_monotonic_s']+r['last_token_s'] for r in successful),default=None)
    span=last-first if first is not None else None
    tokens=sum(r['usage']['completion_tokens'] for r in successful)
    aggregate=sum(max(0,r['usage']['completion_tokens']-1) for r in successful)/span if all_pass and span and span>0 else None
    return {'started_utc':entries[0].get('result',{}).get('start_utc'),
            'concurrency':len(cases),'streams':entries,'passed':all_pass,'wall_s':wall,
            'aggregate_decode_tok_s':aggregate,'aggregate_decode_span_s':span,
            'aggregate_formula':'sum(completion_tokens - 1) / (latest last text SSE - earliest first text SSE)',
            'end_to_end_tok_s':tokens/wall if all_pass and wall>0 else None,
            'per_stream_tok_s':[e.get('result',{}).get('decode_tok_s') for e in entries]}

def run_s4(args, fixture, receipt, save, clock_reader):
    base=args.base.rstrip('/');url=base.rsplit('/v1',1)[0]+'/metrics'
    batches=fixture['s4_batches']
    if len(batches)!=3 or any(len(b)!=4 or any(c['max_tokens']!=512 for c in b) for b in batches):
        raise ValueError('S4 requires three batches of four 512-token streams')
    phase={'name':'decode_s4','kind':'s4','clock_before':clock_reader(),'clock_label':'pending','runs':[]}
    receipt['phases'].append(phase);save()
    try:
        for rep,cases in enumerate(batches,1):
            entry={'rep':rep,'metrics_before':metrics(url)}
            phase['runs'].append(entry)
            entry['batch']=concurrent_batch(base,fixture['model'],cases,getattr(args,'request_timeout',900))
            save()
            if args.metrics_settle:time.sleep(args.metrics_settle)
            entry['metrics_after']=metrics(url)
            entry['mtp']=mtp_delta(entry['metrics_before'],entry['metrics_after'])
            entry['mtp']['attribution']='Whole S4 batch; cannot attribute global counters to an individual concurrent stream.'
            if not entry['batch']['passed']:receipt['passed']=False
            print('decode_s4',rep,json.dumps({k:entry['batch'][k] for k in ('passed','aggregate_decode_tok_s','per_stream_tok_s')}),flush=True)
            save()
    finally:
        phase['clock_after']=clock_reader();phase['clock_label']=clock_label(phase['clock_before'],phase['clock_after'])
        phase['summary']={'aggregate_decode_tok_s':stats([r['batch']['aggregate_decode_tok_s'] for r in phase['runs'] if 'batch' in r]),
                          'per_stream_tok_s':stats([v for r in phase['runs'] if 'batch' in r for v in r['batch']['per_stream_tok_s']])}
        save()

def smoke_checks(result):
    answer=result['text']['content'].strip()
    # A constrained arithmetic sentence is a deterministic, narrow coherence check.
    normalized=' '.join(re.sub(r'[^a-z ]','',answer.lower()).split())
    reasoning=result['text']['reasoning_content']+result['text']['reasoning']
    checks={'nonempty_answer':bool(answer),'coherent_expected_answer':normalized=='two plus two equals four',
            'reasoning_field_present':result['reasoning_field_seen'],'nonempty_reasoning':bool(reasoning.strip()),
            'finished_normally':result['finish_reason']=='stop'}
    return {'expected_answer':'Two plus two equals four.','checks':checks,'passed':all(checks.values()),
            'scope':'Constrained arithmetic/instruction-following check, not a general quality assessment.'}

def run_smoke(args, fixture, receipt, save, clock_reader):
    data={'clock_before':clock_reader(),'clock_label':'pending'}
    receipt['smoke']=data;save()
    try:
        batch=concurrent_batch(args.base,fixture['model'],[fixture['smoke_case']],getattr(args,'request_timeout',180))
        data['request']=batch['streams'][0]
        if 'result' in data['request']:
            data.update(smoke_checks(data['request']['result']))
            data['passed']=data['passed'] and data['request']['passed']
        else:data['passed']=False
        receipt['passed']=data['passed']
    finally:
        data['clock_after']=clock_reader();data['clock_label']=clock_label(data['clock_before'],data['clock_after']);save()

def run_soak(args, fixture, receipt, save, clock_reader):
    duration=getattr(args,'soak_seconds',600)
    data={'target_duration_s':duration,'clock_before':clock_reader(),'clock_label':'pending',
          'batches':[],'errors':[],'prompt_policy':'Reuse the prepared short-chat, exact-8K, and decode pool; cache hits are allowed.',
          'deadline_policy':'Stop starting batches after the duration; let the final batch finish. Request timeout is a socket inactivity timeout.'}
    receipt['soak']=data;save()
    logfile=Path(args.output).with_suffix('.requests.jsonl')
    data['request_log']=logfile.name
    workloads=fixture['soak_cases']
    if len(workloads)!=3:raise ValueError('Soak requires short chat, 8K, and decode cases')
    start=time.perf_counter();batch_index=0;cursor=0;seen_concurrency=set();seen_workloads=set()
    try:
        with logfile.open('w') as log:
            while time.perf_counter()-start<duration:
                concurrency=1+batch_index%4
                cases=[workloads[(cursor+i)%len(workloads)] for i in range(concurrency)]
                cursor+=concurrency
                batch=concurrent_batch(args.base,fixture['model'],cases,getattr(args,'request_timeout',180),keep_events=False)
                batch['rep']=batch_index+1;batch['elapsed_s']=time.perf_counter()-start
                data['batches'].append(batch);seen_concurrency.add(concurrency)
                for entry in batch['streams']:
                    seen_workloads.add(entry['case']['phase'])
                    record={'batch':batch_index+1,'concurrency':concurrency,**entry}
                    log.write(json.dumps(record)+'\n');log.flush()
                    if not entry['passed']:
                        data['errors'].append({'batch':batch_index+1,'stream_index':entry['stream_index'],
                                               'error':entry.get('error'),'warnings':entry.get('result',{}).get('warnings',[])})
                batch_index+=1
                save()
    finally:
        data['duration_s']=time.perf_counter()-start
        data['concurrency_seen']=sorted(seen_concurrency);data['workloads_seen']=sorted(seen_workloads)
        data['request_count']=sum(len(b['streams']) for b in data['batches'])
        data['coverage_complete']=seen_concurrency=={1,2,3,4} and seen_workloads=={c['phase'] for c in workloads}
        data['passed']=not data['errors'] and data['coverage_complete'] and data['duration_s']>=duration
        data['scope']='Client/API stability only; this mode does not inspect kernel logs or establish absence of GPU driver errors.'
        receipt['passed']=data['passed']
        data['clock_after']=clock_reader();data['clock_label']=clock_label(data['clock_before'],data['clock_after']);save()

def extra_markdown(receipt):
    lines=[]
    def fmt(value):return '—' if value is None else f'{value:.2f}'
    for phase in receipt.get('phases',[]):
        if phase.get('kind')!='s4':continue
        lines += ['', '## Decode S4', '', '| Rep | Clock | Aggregate decode tok/s | Per-stream tok/s (0 / 1 / 2 / 3) | Pass |',
                  '|---:|---|---:|---|---|']
        for entry in phase['runs']:
            if 'batch' not in entry:continue
            b=entry['batch']
            lines.append(f"| {entry['rep']} | {phase['clock_label']} | {fmt(b['aggregate_decode_tok_s'])} | {' / '.join(fmt(x) for x in b['per_stream_tok_s'])} | {b['passed']} |")
        lines += ['', 'Aggregate decode = sum(N − 1) divided by the interval from earliest first text SSE to latest last text SSE.',
                  'Each batch has four concurrent 512-token streams. MTP deltas apply to the whole batch.']
    if 'smoke' in receipt:
        s=receipt['smoke'];lines += ['', '## Smoke', '', f"Status: {'PASS' if s.get('passed') else 'FAIL or pending'}; clock: {s['clock_label']}.",
                                  'Expected content: “Two plus two equals four.” Reasoning must be present and nonempty.',
                                  'This is a narrow deterministic check, not a general quality assessment.']
    if 'soak' in receipt:
        s=receipt['soak'];lines += ['', '## Mixed-load soak', '', f"Status: {'PASS' if s.get('passed') else 'FAIL or pending'}; clock: {s['clock_label']}.",
                                  f"Duration: {s.get('duration_s',0):.2f}s; requests: {s.get('request_count',0)}; errors: {len(s['errors'])}.",
                                  f"Concurrency tested: {s.get('concurrency_seen',[])}. Request log: `{s.get('request_log','pending')}`.",
                                  'Client/API stability only; no kernel-log or GPU-driver-error check.']
    if 'passed' in receipt:lines += ['', f"Overall status: {'PASS' if receipt['passed'] else 'FAIL'}."]
    return lines


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('bench','s1','s4','smoke','soak'),default='bench')
    p.add_argument('--soak-seconds',type=float,default=600)
    p.add_argument('--request-timeout',type=float,default=180)
    p.add_argument('--base',default=os.environ.get('BENCH_BASE', 'http://192.168.50.11:18000/v1'))
    p.add_argument('--prompts',default=str(Path(__file__).with_name('prompts.json')))
    p.add_argument('--output',default='receipt.json')
    p.add_argument('--metrics-settle',type=float,default=11)
    p.add_argument('--go',action='store_true',help='Required: explicitly authorize benchmark requests')
    a=p.parse_args()
    if not a.go: p.error('Pass --go only when the server is ready for benchmark requests')
    if a.soak_seconds<=0 or a.request_timeout<=0 or a.metrics_settle<0: p.error('Durations must be positive; metrics settle can be zero')
    receipt=run(a)
    if not receipt['passed']: raise SystemExit(1)
