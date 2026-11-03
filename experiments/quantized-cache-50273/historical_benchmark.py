import os, sys, json, time, hashlib, statistics, traceback
from datetime import datetime, timezone
from pathlib import Path
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, QuantizedCache

repo = Path(os.environ['BENCH_REPO']).resolve()
outdir = Path(os.environ['BENCH_OUT']); outdir.mkdir(parents=True,exist_ok=True)
assert Path(transformers.__file__).resolve().is_relative_to(repo / 'src'), (
    'Wrong Transformers import: ' + transformers.__file__)
assert torch.cuda.device_count() >= 2, 'Requires two NVIDIA GPUs'
TARGET = os.getenv('BENCH_TARGET', 'Qwen/Qwen2.5-1.5B-Instruct')
ASSISTANT = 'Qwen/Qwen2.5-0.5B-Instruct'
LENGTHS = (1024, 4096)
REPEATS = 2
NEW_TOKENS = 48
MODES = ('dynamic_greedy', 'dynamic_assisted', 'fp8_greedy', 'fp8_assisted')
MiB = 1024 * 1024
torch.manual_seed(127)

tok = AutoTokenizer.from_pretrained(TARGET)
target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float16).to('cuda:0').eval()
assistant = AutoModelForCausalLM.from_pretrained(ASSISTANT, dtype=torch.float16).to('cuda:1').eval()
for model in (target, assistant):
    model.generation_config.cache_implementation = None
    model.generation_config.cache_config = None
assistant.generation_config.num_assistant_tokens = 4
assistant.generation_config.num_assistant_tokens_schedule = 'constant'
assistant.generation_config.assistant_confidence_threshold = 0.0

doc_path = repo / 'docs/source/en/kv_cache.md'
doc = doc_path.read_text()
doc_ids = tok.encode(doc, add_special_tokens=False)

def make_inputs(length):
    extract = tok.decode(doc_ids[:max(1, length-225)], skip_special_tokens=False)
    content = ('Answer the following question using this real Transformers documentation excerpt. '
               'If the excerpt lacks the answer, state that clearly.\n\nDOCUMENT:\n' + extract +
               '\n\nQUESTION: Explain the trade-off between quantized KV-cache memory use '
               'and inference latency in two sentences.')
    return tok.apply_chat_template([{'role':'user','content':content}], tokenize=True,
                                   add_generation_prompt=True, return_dict=True,
                                   return_tensors='pt').to('cuda:0')

def nbytes(x):
    if torch.is_tensor(x): return x.numel() * x.element_size()
    if isinstance(x, (tuple,list)): return sum(nbytes(y) for y in x)
    return 0

def cache_bytes(cache):
    return sum(nbytes(getattr(layer, attr, None)) for layer in cache.layers
               for attr in ('keys','values','_quantized_keys','_quantized_values'))

def run_case(mode, inputs, new_tokens=NEW_TOKENS, force_id=None):
    quant = mode.startswith('fp8')
    assisted = mode.endswith('assisted')
    cache = QuantizedCache(backend='fp8',config=target.config) if quant else DynamicCache(config=target.config)
    hooks = []; crop_events=[]; forced_calls=[0]
    if force_id is not None:
        assert assisted
        def force_token(_module, _args, logits):
            forced_calls[0] += 1
            x = torch.full_like(logits, -10000.0)
            x[...,force_id] = 10000.0
            return x
        hooks.append(assistant.get_output_embeddings().register_forward_hook(force_token))
        original_crop = cache.crop
        def record_crop(n):
            before = cache.get_seq_length()
            result = original_crop(n)
            if n < 0: crop_events.append([n,before,cache.get_seq_length()])
            return result
        cache.crop = record_crop
    torch.cuda.empty_cache()
    baseline = [torch.cuda.memory_allocated(i) for i in (0,1)]
    for i in (0,1): torch.cuda.reset_peak_memory_stats(i)
    t0 = time.perf_counter()
    result = {'mode': mode, 'prompt_tokens':inputs['input_ids'].shape[1], 'forced':force_id is not None}
    try:
        with torch.inference_mode():
            output = target.generate(**inputs, past_key_values=cache,
                assistant_model=assistant if assisted else None,
                do_sample=False, max_new_tokens=new_tokens,
                use_cache=True, disable_compile=True)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        duration = time.perf_counter()-t0
        token_ids = output[0,inputs['input_ids'].shape[1]:].tolist()
        result.update(status='PASS', tokens=token_ids, generated=len(token_ids),
            latency_s=round(duration,3), tokens_per_s=round(len(token_ids)/duration,3),
            kv_MiB=round(cache_bytes(cache)/MiB,2),
            peak_alloc_MiB=[round(torch.cuda.max_memory_allocated(i)/MiB,2) for i in (0,1)],
            incremental_peak_MiB=[round((torch.cuda.max_memory_allocated(i)-baseline[i])/MiB,2) for i in (0,1)],
            crop_events=crop_events[:10], forced_head_calls=forced_calls[0])
    except Exception as exc:
        result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc()[-2500:])
    finally:
        for h in hooks: h.remove()
        del cache
        torch.cuda.empty_cache()
    return result

print('Warmup...')
warmup = make_inputs(128)
for mode in MODES:
    w = run_case(mode,warmup,4)
    print('Warmup',mode,w['status'],w.get('error','')[:130],flush=True)
    if w['status'] != 'PASS':
        (outdir/'results.json').write_text(json.dumps({'warmup_failure':w},indent=2))
        raise RuntimeError('Warmup failed; review results.json before benchmarking')
del warmup

results=[]; baseline_tokens={}
for length in LENGTHS:
    inputs=make_inputs(length)
    for trial in range(REPEATS):
        for mode in MODES:
            r=run_case(mode,inputs)
            r.update(length_target=length,trial=trial+1)
            results.append(r)
            print(f"{length=} trial={trial+1} {mode}: {r['status']}  "
                  f"{r.get('latency_s', '-')}s  {r.get('tokens_per_s','-')} tok/s",flush=True)
            if r['status']=='PASS' and mode.endswith('greedy') and trial==0 and length==LENGTHS[0]:
                baseline_tokens[mode] = r['tokens']
    del inputs

# Independently verify that forcing a wrong draft triggers an actual rollback.
short=make_inputs(LENGTHS[0]); refs={}
for mode in ('dynamic_greedy','fp8_greedy'):
    if mode in baseline_tokens: refs[mode] = baseline_tokens[mode]
first_tokens={t[0] for t in refs.values() if t}
force_id=next(t for t in (10,11,12,13,14,15,16) if t not in first_tokens and t != tok.eos_token_id)
forced={}
for mode in ('dynamic_assisted','fp8_assisted'):
    r=run_case(mode,short,force_id=force_id)
    key='fp8_greedy' if mode.startswith('fp8') else 'dynamic_greedy'
    r['parity_with_own_greedy']=(r.get('tokens')==refs.get(key))
    r['negative_crop_seen']=bool(r.get('crop_events'))
    forced[mode]=r
    print('Forced rejection:',mode,r['status'],'rollback=',r['negative_crop_seen'],
          'greedy parity=',r['parity_with_own_greedy'],flush=True)

for r in results:
    if r['mode'].endswith('assisted'):
        baseline_mode=r['mode'].replace('assisted','greedy')
        base=next((x for x in results if x['length_target']==r['length_target']
                   and x['trial']==r['trial'] and x['mode']==baseline_mode),None)
        r['parity_with_own_greedy']=(r.get('status')=='PASS' and base is not None
                                    and r['tokens']==base.get('tokens'))

valid=(all(r['status']=='PASS' for r in results)
       and all(r.get('parity_with_own_greedy',True) for r in results)
       and all(r['status']=='PASS' and r['negative_crop_seen'] and
               r['parity_with_own_greedy'] and r['forced_head_calls']>0 for r in forced.values()))
meta=dict(datetime_utc=datetime.now(timezone.utc).isoformat(),
          transformers_version=transformers.__version__,
          torch_version=torch.__version__, python=sys.version.split()[0],
          GPUs=[torch.cuda.get_device_name(i) for i in (0,1)],
          target=TARGET,assistant=ASSISTANT, lengths=LENGTHS, repeats=REPEATS,
          max_new_tokens=NEW_TOKENS, repo_commit=os.environ['BENCH_SHA'],
          document_path='docs/source/en/kv_cache.md',
          document_sha256=hashlib.sha256(doc.encode()).hexdigest(),
          valid=valid)
raw={'metadata':meta,'runs':results,'forced_rejection':forced}
(outdir/'results.json').write_text(json.dumps(raw,indent=2))

lines=['# Quantized KV-cache + assisted decoding: reproducible benchmark',
       '',f"**Outcome: {'PASS' if valid else 'INCOMPLETE / FAIL'}** (see raw JSON)",
       '',f"- PR: https://github.com/huggingface/transformers/pull/50277",
       f"- Issue: https://github.com/huggingface/transformers/issues/50273",
       f"- Transformers fork commit: `{meta['repo_commit']}`",
       f"- Models: `{TARGET}` target on GPU 0; `{ASSISTANT}` assistant on GPU 1",
       f"- GPUs: {meta['GPUs']} | PyTorch {meta['torch_version']}",
       f"- Workload: actual Transformers KV-cache documentation excerpt; {REPEATS} repetitions, {NEW_TOKENS} max new tokens",
       '- Latencies include prefill; tokens/s are **end-to-end**, not isolated decode speed.',
       '- Memory is PyTorch CUDA peak *allocated*, not process-resident VRAM; KV MiB is measured from cache tensor storage.',
       '', '| Context target | Mode | Successful runs | Median latency (s) | Median tokens/s | Median KV MiB | Peak GPU 0 MiB |',
       '|---:|---|---:|---:|---:|---:|---:|']
for length in LENGTHS:
    for mode in MODES:
        rs=[r for r in results if r['length_target']==length and r['mode']==mode]
        ok=[r for r in rs if r['status']=='PASS']
        median=lambda k: f"{statistics.median(r[k] for r in ok):.2f}" if ok else '—'
        lines.append(f'| {length} | {mode} | {len(ok)}/{len(rs)} | {median("latency_s")} | '
                     f'{median("tokens_per_s")} | {median("kv_MiB")} | '
                     f'{statistics.median(r["peak_alloc_MiB"][0] for r in ok):.2f} |' if ok else
                     f'| {length} | {mode} | 0/{len(rs)} | — | — | — | — |')
    for base,assisted in (('dynamic_greedy','dynamic_assisted'),('fp8_greedy','fp8_assisted')):
        b=[r['latency_s'] for r in results if r['length_target']==length and r['mode']==base and r['status']=='PASS']
        a=[r['latency_s'] for r in results if r['length_target']==length and r['mode']==assisted and r['status']=='PASS']
        if a and b:
            lines.append(f'- Context {length}, `{assisted}` speed ratio vs `{base}`: '
                         f'**{statistics.median(b)/statistics.median(a):.2f}×** '
                         '(values below 1× mean slower).')
lines.extend(['','## Correctness / forced rejection',
              '| Check | Status | Negative crop observed | Exact token parity with matching greedy mode |',
              '|---|---|---|---|'])
for mode,r in forced.items():
    lines.append(f'| {mode} | {r["status"]} | {r["negative_crop_seen"]} | {r["parity_with_own_greedy"]} |')
lines.extend(['', '- Assisted greedy comparisons are made **against the same cache precision**; FP8 and Dynamic output need not match each other.',
              '- A completed run is not proof of speedup. The values above determine whether the feature offers practical benefit.',
              '- The benchmark uses public documentation, not private workloads. It does not establish production demand or generalize to other models.',
              '- If errors, mismatches, or missing rollbacks appear, the result is **not** evidence of a validated feature.',
              '', '## How to reproduce',
              'Use two NVIDIA GPUs, install dependencies compatible with the tested PyTorch version, '
              'checkout the fork commit above, then run `benchmark.py` with '
              '`BENCH_REPO` pointing at that checkout and `BENCH_OUT` set to a writable directory. '
              'The runner imports Transformers from `BENCH_REPO/src` and asserts that source path.',
              '', 'Detailed per-run timings, token sequences, memory measurements, and errors are in `results.json`.'])
(outdir/'README.md').write_text('\n'.join(lines)+'\n')
print('BENCHMARK_VALID=',valid)
print('Output files:',outdir)
