#!/usr/bin/env python3
"""CPU-only fixture creation from the three pinned Hugging Face tokenizer files."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import uuid
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
REV = 'bc246cc2848e17b37f377eda8fd825c5b3018d1f'

def prepare():
    hf = AutoTokenizer.from_pretrained(ROOT / 'tokenizer', local_files_only=True)
    tok = Tokenizer.from_file(str(ROOT / 'tokenizer/tokenizer.json'))
    def count(messages):
        rendered = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tok.encode(rendered, add_special_tokens=False).ids)
    cases = []
    paragraph = ' The engineer checks the instrument, records the measurement, and compares the result with the previous observation.'
    for size in (8192, 32768):
        for rep in range(4):
            prefix = uuid.uuid4().hex + '\nRead the following reference text. Reply with one word.\n'
            content = prefix + paragraph * (size // 24)
            while count([{'role':'user','content':content}]) > size:
                content = content[:-len(paragraph)]
            messages = [{'role':'user', 'content': content}]
            gap = size - count(messages)
            messages[0]['content'] += ' x' * gap
            assert count(messages) == size
            cases.append({'phase':f'prefill_{size}', 'rep':rep, 'warmup':rep==0,
                          'prompt_tokens_local':count(messages), 'messages':messages, 'max_tokens':1})
    prompts = {
        'prose': 'Write a detailed, continuous technical essay of at least 2000 words about how municipal water treatment works. Explain each stage in order. Start the essay now and do not summarize early.',
        'code': 'Write a complete Python implementation of a persistent key-value store with a write-ahead log, recovery, checksums, compaction, and extensive unit tests. Produce at least 2000 words of code. Start with the imports and continue the implementation.',
        'json': 'Return a single JSON object with a records array of 200 distinct inventory records. Each record must contain id, name, category, quantity, price, warehouse, and an array of three tags. Return only JSON and write all 200 records.'}
    for kind, prompt in prompts.items():
        for rep in range(1,4):
            messages = [{'role':'user','content':uuid.uuid4().hex + '\n' + prompt}]
            cases.append({'phase':f'decode_{kind}', 'rep':rep, 'warmup':False,
                          'prompt_tokens_local':count(messages), 'messages':messages, 'max_tokens':512})
    fixture = {'model':'step5','tokenizer_revision':REV,
               'source':'https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4/tree/'+REV,
               'tokenizer_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'tokenizer').iterdir())},
               'cases':cases}
    (ROOT/'prompts.json').write_text(json.dumps(fixture,indent=2)+'\n')
    print('Prepared',len(cases),'requests;',[(x['phase'],x['rep'],x['prompt_tokens_local']) for x in cases])
    return fixture

if __name__ == '__main__': prepare()
