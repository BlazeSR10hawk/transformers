# SINGLE STANDALONE KAGGLE CELL: clone the CURRENT PR branch and reproduce
# irreversible FP8 scale effects caused by slicing previously quantized states.
# Does NOT edit or push any repository branch.
import os, sys, subprocess, pathlib

REPO = pathlib.Path('/kaggle/working/quantized-cache-50273-scale-check')
BRANCH = 'fix/quantized-cache-crop-50273'
remote = 'https://github.com/BlazeSR10hawk/transformers.git'
if not (REPO / '.git').exists():
    subprocess.run(['git', 'clone', '-q', '--depth=1', '--branch', BRANCH, remote, str(REPO)], check=True)
else:
    subprocess.run(['git', 'fetch', '-q', '--depth=1', 'origin', BRANCH], cwd=REPO, check=True)
    subprocess.run(['git', 'checkout', '-q', '--detach', 'FETCH_HEAD'], cwd=REPO, check=True)

AUDIT = r'''import torch
from transformers import LlamaConfig, QuantizedCache
cfg = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=512,
                  num_hidden_layers=1, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
prefix = torch.full((1, 2, 8, 64), 0.01, dtype=torch.float16, device=device)
rejected = torch.full((1, 2, 4, 64), 10000., dtype=torch.float16, device=device)
reference = QuantizedCache(backend='fp8', config=cfg)
speculative = QuantizedCache(backend='fp8', config=cfg)
reference.update(prefix, prefix, 0)
speculative.update(torch.cat((prefix, rejected), -2),
                   torch.cat((prefix, rejected), -2), 0)

# Simulate the superseded PR's FP8 tensor-slicing rollback. Do NOT call the
# current crop() method: the current PR intentionally rejects this operation.
layer = speculative.layers[0]
for name in ('_quantized_keys', '_quantized_values'):
    q, scale = getattr(layer, name)
    setattr(layer, name, (q[..., :8, :], scale))
layer.cumulative_length = 8
ref_layer = reference.layers[0]
a = ref_layer._dequantize(ref_layer._quantized_keys)
b = layer._dequantize(layer._quantized_keys)
err = (a.float() - b.float()).abs().max().item()
print('Reference length:', reference.get_seq_length())
print('Simulated rollback length:', speculative.get_seq_length())
print('Reference scale:', ref_layer._quantized_keys[1].item())
print('Rollback scale:', layer._quantized_keys[1].item())
print('Reference retained KV mean:', a.float().mean().item())
print('Simulated rollback KV mean:', b.float().mean().item())
print('Max retained KV error:', err)
print('Identical retained KV:', torch.equal(a, b))
assert reference.get_seq_length() == speculative.get_seq_length() == 8
assert err > 0 and not torch.equal(a, b)
print('PASS: former FP8 slice-based rollback irreversibly loses retained values')
assert speculative.is_croppable is False
try:
    speculative.crop(-2)
except ValueError as e:
    assert 'does not support rollback' in str(e)
    print('PASS: current PR explicitly rejects unsupported rollback')
else:
    raise AssertionError('Expected current PR to reject quantized cache rollback')
'''
env = os.environ.copy()
env['PYTHONPATH'] = str(REPO / 'src') + os.pathsep + env.get('PYTHONPATH', '')
print('PR code version:', subprocess.check_output(['git', 'rev-parse', '--short=12', 'HEAD'], cwd=REPO, text=True).strip())
subprocess.run([sys.executable, '-c', AUDIT], env=env, check=True)
