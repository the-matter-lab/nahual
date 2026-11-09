# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browse data splits and evaluation samples as single-turn chats, and prompt a checkpoint.

    python scripts/viewer/server.py --samples hub/samples

Then open http://localhost:8000 (on a remote machine, forward the port with
ssh -L 8000:localhost:8000). Data splits come from NAHUAL_RAW_DIR (python -m nahual.prepare
download). Samples are the predictions.pt files under the --samples folders (default
evals/results). Checkpoints are found next to them (hub/checkpoints/<model>/seed<k>/) and in
logs/; Nahual seed 0, or --checkpoint, loads in the background at startup. The server
starts with the standard library only: NumPy and pyarrow load with the first example, and
PyTorch with the model, which needs a GPU.
"""
import argparse
from collections import Counter, OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import pickle
import re
import sys
import threading
from urllib.parse import parse_qs, urlparse
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from nahual.paths import raw_dir  # noqa: E402  (standard library only)

STATIC = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
          '/style.css': ('style.css', 'text/css')}
SYMBOLS = ('X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br '
           'Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm '
           'Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr '
           'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
LATTICE_ORIGIN = 119  # atom 119 marks the cell origin; the next three rows are the cell vectors
SPECIAL = re.compile(r'<\|[a-z_]+\|>')  # chat markers such as the <|im_end|> stop token
STOP_MARK = '<|im_end|>'  # a response ends here; batched generation keeps sampling after it
ROLES = ('train', 'validation', 'test')
DEFAULT_SOURCE = 'data:nciatlas_finetune_train'  # shown first, if present


class XYZ:
    """Stand-in for nahual.tokens.XYZ when reading predictions.pt (importing it pulls in transformers)."""

    def __init__(self, atoms=None, coords=None):
        self.atoms, self.coords = atoms, coords

    def __setstate__(self, state):
        self.__dict__.update(state)


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return XYZ if (module, name) == ('nahual.tokens', 'XYZ') else super().find_class(module, name)

    def persistent_load(self, pid):  # a tensor: leave the file to torch.load
        raise pickle.UnpicklingError('tensor storage')


def load_predictions(path):
    """predictions.pt without PyTorch (a torch zip archive whose pickle holds no tensors)."""
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith('data.pkl'))
            with archive.open(name) as f:
                return Unpickler(f).load()
    except (zipfile.BadZipFile, StopIteration, pickle.UnpicklingError):
        import torch
        return torch.load(path, map_location='cpu', weights_only=False)


def formula(atoms):
    """Hill formula: C, then H, then the other elements alphabetically (all alphabetically without C)."""
    counts = Counter(SYMBOLS[int(a)] for a in atoms)
    first = ['C', 'H'] if 'C' in counts else []
    order = [e for e in first if e in counts] + sorted(set(counts) - set(first))
    return ''.join(e + (str(counts[e]) if counts[e] > 1 else '') for e in order)


def vacuum_axis(coords, vectors, min_atoms=10):
    """The axis of a slab's vacuum gap (the atoms span much less of it than of the other axes), or None."""
    import numpy as np
    if len(coords) < min_atoms or abs(np.linalg.det(vectors)) < 1e-6:
        return None
    frac = np.sort(np.linalg.solve(vectors.T, coords.T) % 1.0, axis=1)
    gaps = np.diff(np.concatenate([frac, frac[:, :1] + 1], axis=1), axis=1).max(axis=1)  # largest periodic gap
    span = 1 - gaps
    first, second = np.argsort(span)[:2]
    return int(first) if span[first] <= 0.7 and span[second] - span[first] >= 0.15 else None


def segment(item):
    """A text or structure segment (str, XYZ, or a data-row / generation dict) as page JSON; None if empty."""
    import numpy as np
    if isinstance(item, dict):
        item = XYZ(item['atoms'], item['coords']) if item.get('type') == 'xyz' else item.get('text') or ''
    if not isinstance(item, XYZ) and not hasattr(item, 'atoms'):
        text = SPECIAL.sub('', str(item)).strip()
        return {'type': 'text', 'text': text} if text else None
    atoms = np.asarray(item.atoms).astype(int)
    coords = np.asarray(item.coords, dtype=float).reshape(-1, 3)
    out = {'type': 'xyz', 'atoms': atoms.tolist(), 'coords': coords.round(5).tolist(), 'cell': None}
    if len(atoms) >= 4 and atoms[0] == LATTICE_ORIGIN:
        origin, vectors = coords[0], coords[1:4] - coords[0]
        out['cell'] = {'origin': origin.tolist(), 'vectors': vectors.tolist(),
                       'vacuum': vacuum_axis(coords[4:] - origin, vectors)}
        atoms, coords = atoms[4:], coords[4:]
    keep = (atoms >= 1) & (atoms <= 118)
    lines = [str(int(keep.sum())), '']
    lines += [f'{SYMBOLS[a]} {x:.4f} {y:.4f} {z:.4f}' for a, (x, y, z) in zip(atoms[keep], coords[keep])]
    out['xyz'] = '\n'.join(lines)
    out['formula'] = formula(atoms[keep])
    return out


def segments(items):
    """Segments up to the first stop marker."""
    out = []
    for item in items or []:
        stop = isinstance(item, str) and STOP_MARK in item
        s = segment(item.split(STOP_MARK, 1)[0] if stop else item)
        if s:
            out.append(s)
        if stop:
            break
    return out


def title(items, length=160):
    """One line for the example list: the prompt's text with each structure as its formula."""
    parts = []
    for item in items or []:
        s = segment(item)
        if s:
            parts.append(s['formula'] if s['type'] == 'xyz' else ' '.join(s['text'].split()))
    text = ' '.join(parts) or '(empty)'
    return text if len(text) <= length else text[:length - 1] + '…'


def message(segs):
    """A prompt as message-box text: its text, and each structure as an ```xyz block (extended XYZ
    Lattice/Origin/pbc for a cell)."""
    parts = []
    for s in segs:
        if s['type'] == 'text':
            parts.append(s['text'])
            continue
        atoms, coords, comment = s['atoms'], s['coords'], ''
        if s['cell']:
            numbers = lambda values: ' '.join(f'{x:.5f}' for x in values)
            pbc = ' '.join('F' if a >= 123 else 'T' for a in atoms[1:4])  # 123-125 are open (non-periodic) vectors
            comment = (f'Lattice="{numbers(x for v in s["cell"]["vectors"] for x in v)}" '
                       f'Origin="{numbers(s["cell"]["origin"])}" pbc="{pbc}"')
            atoms, coords = atoms[4:], coords[4:]
        lines = [str(len(atoms)), comment] + [f'{SYMBOLS[a]} {x:.5f} {y:.5f} {z:.5f}' for a, (x, y, z) in zip(atoms, coords)]
        parts.append('```xyz\n' + '\n'.join(lines) + '\n```')
    return '\n'.join(parts)


def parse_xyz(block):
    """One ```xyz block: an atom count, a comment line, then 'symbol x y z' rows."""
    lines = block.strip('\n').split('\n')
    try:
        n = int(lines[0].split()[0])
        rows = [line.split() for line in lines[2:] if line.strip()]
        if len(rows) != n:
            raise ValueError(f'the block lists {n} atoms but has {len(rows)} atom rows')
        atoms = [int(r[0]) if r[0].isdigit() else SYMBOLS.index(r[0].capitalize()) for r in rows]
        coords = [[float(x) for x in r[1:4]] for r in rows]
    except (IndexError, ValueError) as exc:
        raise ValueError(f'Cannot read an xyz block: {exc}') from None
    comment = lines[1] if len(lines) > 1 else ''
    field = lambda key: re.search(rf'{key}="([^"]*)"', comment)
    if field('Lattice'):
        v = [float(x) for x in field('Lattice').group(1).split()]
        origin = [float(x) for x in field('Origin').group(1).split()] if field('Origin') else [0.0, 0.0, 0.0]
        pbc = field('pbc').group(1).split() if field('pbc') else ['T'] * 3
        atoms = [LATTICE_ORIGIN] + [(120 if pbc[i] in ('T', 'True', '1') else 123) + i for i in range(3)] + atoms
        coords = [origin] + [[o + x for o, x in zip(origin, v[3 * i:3 * i + 3])] for i in range(3)] + coords
    return {'type': 'xyz', 'atoms': atoms, 'coords': coords}


def parse_message(text):
    """Message-box text as prompt segments: ```xyz blocks become structures, the rest text."""
    out, pos = [], 0
    for m in re.finditer(r'```xyz\n(.*?)```', text, re.S):
        before = text[pos:m.start()].removesuffix('\n')
        before = before.removeprefix('\n') if pos else before
        if before.strip():
            out.append({'type': 'text', 'text': before})
        out.append(parse_xyz(m.group(1)))
        pos = m.end()
    rest = text[pos:].removeprefix('\n') if pos else text[pos:]
    if rest.strip():
        out.append({'type': 'text', 'text': rest})
    return out


def score_of(sample):
    try:
        score = float((sample.get('metric') or {}).get('score'))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


RMSD_TASKS = {'t1x', 'nci', 'lnqm', 'qcg_1to10'}
ORACLE_UNITS = {'gap': 'meV', 'homo': 'meV', 'lumo': 'meV', 'mu': 'D', 'alpha': 'Bohr³', 'Cv': 'cal/(mol·K)'}
NUMBER = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def describe(task, sample, reference):
    """A sample's evaluation metric, named: 'RMSD to reference 0.153 Å', 'no match', ..."""
    metric = sample.get('metric') or {}
    info, score = metric.get('info') or {}, score_of(sample)
    task = task.removesuffix('_xtb_relaxed_crude')
    if sample.get('egnn_pred') is not None:  # conditional QM9 generation
        unit = ORACLE_UNITS.get(task.rsplit('_', 1)[-1], '')
        text = f"EGNN oracle property {sample['egnn_pred']:.5g} {unit}"
        if sample.get('self_pred') is not None:
            text += f" · self-predicted {sample['self_pred']:.5g} {unit}"
        return text
    if metric.get('valid') is False:
        return 'not scored: ' + (info.get('msg') or 'invalid')
    if task == 'csp':
        return (f"matches the reference structure, normalized RMS distance {info['rms']:.3f}"
                if info.get('match') else 'does not match the reference structure')
    if 'ads_rmsd' in info:  # adsorption on surfaces and in MOFs
        return f"adsorbate RMSD {info['ads_rmsd']:.3f} Å · full RMSD {info['full_rmsd']:.3f} Å"
    if 'pred_rmsd' in info:  # NEB
        text = f"mean RMSD over the frames {info['pred_rmsd']:.3f} Å"
        if math.isfinite(info.get('pred_ads_rmsd', math.inf)):
            text += f" · adsorbate {info['pred_ads_rmsd']:.3f} Å"
        return text
    if task == 'geomconf_diverse':
        return 'scored as a set: coverage and matching RMSD of all samples'
    if task in RMSD_TASKS and score is not None:
        text = f'RMSD to reference {score:.3f} Å'
        if 'same_connectivity' in info:
            text += ' · same connectivity' if info['same_connectivity'] else ' · different connectivity'
        if sample.get('pred_eform') is not None:
            text += f" · xTB formation energy {sample['pred_eform']:.4f} Ha"
        return text
    # Property prediction is scored by the MAE over the test set; show this sample's absolute error.
    said = lambda items: ' '.join(x if isinstance(x, str) else '' for x in items or [])
    predicted, true = NUMBER.search(said(sample.get('prediction'))), NUMBER.search(said(reference))
    if predicted and true:
        unit = said(reference)[true.end():].split()
        return f"absolute error {abs(float(predicted.group()) - float(true.group())):.4g} {unit[0] if unit else ''}".strip()
    return f'score {score:.4g}' if score is not None else ''


def task_names():
    """Benchmark task names, read from evals/tasks/*/task.json without importing evals."""
    names = set()
    for manifest in (ROOT / 'evals/tasks').glob('*/task.json'):
        names.update(task['name'] for task in json.loads(manifest.read_text())['tasks'])
    return names


def is_task(name, tasks):
    return name in tasks or name.removesuffix('_xtb_relaxed_crude') in tasks


def sample_runs(roots):
    """{run: {task: predictions.pt}} for <root>/[<run>/...]<task>/ and <root>/<task>/<run>/ layouts."""
    tasks = task_names()
    runs = {}
    for root in roots:
        paths = [p for pattern in ('*/predictions.pt', '*/*/predictions.pt', '*/*/*/predictions.pt')
                 for p in root.glob(pattern)]
        for path in sorted(paths):
            parts = path.relative_to(root).parts[:-1]
            if is_task(parts[-1], tasks):  # samples/<model>/seed<k>/<task>/
                run, task = '/'.join(parts[:-1]) or root.name, parts[-1]
            elif is_task(parts[0], tasks):  # evals/results/<task>/<run>/
                run, task = '/'.join(parts[1:]), parts[0]
            else:
                continue
            if task in runs.get(run, {}):
                run = f'{root.name}/{run}'
            runs.setdefault(run, {})[task] = str(path)
    return runs


class Split:
    """Rows of a data split, read from its parquet files one row group at a time."""

    def __init__(self, files):
        import pyarrow.parquet as pq
        self.groups, self.tables, start = [], OrderedDict(), 0
        for path in files:
            parquet = pq.ParquetFile(path)
            for g in range(parquet.metadata.num_row_groups):
                rows = parquet.metadata.row_group(g).num_rows
                self.groups.append((parquet, g, start, rows))
                start += rows
        self.count = start

    def rows(self, start, stop, columns=('instruction', 'response')):
        out = []
        for i, (parquet, g, first, rows) in enumerate(self.groups):
            if first + rows <= start or first >= stop:
                continue
            if (i, columns) not in self.tables:
                if len(self.tables) >= 4:
                    self.tables.popitem(last=False)
                self.tables[i, columns] = parquet.read_row_group(g, columns=list(columns))
            lo = max(start, first) - first
            out += self.tables[i, columns].slice(lo, min(stop, first + rows) - first - lo).to_pylist()
        # the grouped ODAC test split stores its segment lists as JSON strings
        return [{k: json.loads(v) if isinstance(v, str) else v for k, v in row.items()} for row in out]

    def titles(self, start, stop):
        return [title(row['instruction']) for row in self.rows(start, stop, ('instruction',))]

    def example(self, index):
        row = self.rows(index, index + 1)[0]
        return {'prompt': segments(row['instruction']), 'references': [segments(row['response'])], 'samples': []}


class Samples:
    """The entries of one predictions.pt of a task."""

    def __init__(self, path, task):
        self.entries, self.task = load_predictions(path), task
        self.count = len(self.entries)

    def titles(self, start, stop):
        return [title(e['prompt']) for e in self.entries[start:stop]]

    def example(self, index):
        entry = self.entries[index]
        truth = entry['ground_truth']  # a response, or {'responses': [...]} for tasks with several references
        truths = truth['responses'] if isinstance(truth, dict) else [truth]
        return {'prompt': segments(entry['prompt']), 'references': [segments(r) for r in truths],
                'samples': [{'segments': segments(s.get('prediction')), 'note': describe(self.task, s, truths[0])}
                            for s in entry['samples']]}


def find_checkpoints(roots):
    """{path: label}: released checkpoints next to the samples (<hub>/checkpoints/<model>/seed<k>/)
    and the newest checkpoint of each training run in logs/."""
    found = {}
    for hub in sorted({root.parent for root in roots} | {ROOT / 'hub'}):
        for path in sorted(hub.glob('checkpoints/*/*/step_*.pt')):
            found[str(path)] = '/'.join(path.relative_to(hub / 'checkpoints').parts[:-1])
    for run in sorted(ROOT.glob('logs/*/*/checkpoints')):
        steps = [p for p in run.glob('step_*.pt') if p.stem[5:].isdigit()]
        if steps:
            newest = max(steps, key=lambda p: int(p.stem[5:]))
            found[str(newest)] = f'{run.parent.name} ({newest.stem})'
    return found


class Store:
    """Data splits, sample files and one model, shared by all requests."""

    def __init__(self, sample_roots, checkpoint=None, device='cuda', keep=3):
        self.roots = [Path(r).resolve() for r in sample_roots if Path(r).exists()]
        self.checkpoints = find_checkpoints(self.roots)
        if checkpoint:
            self.checkpoints.setdefault(str(checkpoint), str(checkpoint))
        released = [p for p, label in self.checkpoints.items() if label == 'nahual/seed0']
        self.default = str(checkpoint) if checkpoint else released[0] if released else None
        self.device, self.keep = device, keep
        self.opened, self.lock, self.loading = OrderedDict(), threading.Lock(), {}
        self.model_lock, self.model, self.model_path = threading.Lock(), None, None
        self.state_lock, self.target, self.state, self.error = threading.Lock(), None, 'none', ''

    def splits(self):
        manifest = raw_dir / 'manifest.json'
        return json.loads(manifest.read_text())['splits'] if manifest.exists() else {}

    def sources(self):
        splits = self.splits()
        out = [{'id': 'data:' + name, 'group': 'Data · ' + role, 'label': name, 'count': splits[name].get('rows')}
               for role in ROLES for name in sorted(splits) if splits[name]['roles'][0] == role]
        out += [{'id': 'samples:' + path, 'group': 'Samples · ' + run, 'label': task, 'count': None}
                for run, tasks in sorted(sample_runs(self.roots).items()) for task, path in sorted(tasks.items())]
        ids = [source['id'] for source in out]
        return {'sources': out, 'source': DEFAULT_SOURCE if DEFAULT_SOURCE in ids else ids[0] if ids else None,
                'checkpoints': [{'path': p, 'label': label} for p, label in self.checkpoints.items()],
                'default': self.default}

    def open(self, source):
        """The Split or Samples behind a source id, loaded once (one loader per source) and kept LRU."""
        with self.lock:
            if source in self.opened:
                self.opened.move_to_end(source)
                return self.opened[source]
            loading = self.loading.setdefault(source, threading.Lock())
        with loading:
            with self.lock:
                if source in self.opened:
                    return self.opened[source]
            kind, _, name = source.partition(':')
            if kind == 'data':
                opened = Split([raw_dir / f for f in self.splits()[name]['files']])
            elif kind == 'samples':
                path = Path(name).resolve()
                if not any(path.is_relative_to(root) for root in self.roots):
                    raise PermissionError(f'{name} is not under a --samples folder')
                task = path.parent.name if is_task(path.parent.name, task_names()) else path.parent.parent.name
                opened = Samples(path, task)
            else:
                raise ValueError(f'Unknown source {source!r}')
            with self.lock:
                self.opened[source] = opened
                while len(self.opened) > self.keep:
                    self.opened.popitem(last=False)
                self.loading.pop(source, None)
            return opened

    def items(self, source, offset, limit):
        opened = self.open(source)
        offset = min(max(offset, 0), opened.count)
        stop = min(offset + limit, opened.count)
        return {'count': opened.count, 'offset': offset, 'titles': opened.titles(offset, stop)}

    def example(self, source, index):
        opened = self.open(source)
        index = min(max(index, 0), opened.count - 1)
        out = {'count': opened.count, 'index': index, **opened.example(index)}
        out['message'] = message(out['prompt'])
        return out

    def load_model(self, checkpoint):
        with self.model_lock:
            if self.model_path != checkpoint:
                import torch
                from evals.backends.checkpoint import load_checkpoint_model
                if self.device.startswith('cuda') and not torch.cuda.is_available():
                    raise RuntimeError('no GPU here; generation needs one')
                self.model = self.model_path = None
                torch.cuda.empty_cache()
                self.model = load_checkpoint_model(checkpoint, device=self.device)
                self.model_path = checkpoint
            return self.model

    def model_status(self):
        with self.state_lock:
            return {'checkpoint': self.target, 'state': self.state, 'error': self.error}

    def select_model(self, checkpoint):
        """Load a checkpoint in the background, so it is ready when a message is sent."""
        with self.state_lock:
            if checkpoint == self.target and self.state in ('loading', 'ready'):
                return {'checkpoint': self.target, 'state': self.state, 'error': self.error}
            self.target, self.state, self.error = checkpoint, 'loading', ''
        threading.Thread(target=self._load, args=(checkpoint,), daemon=True).start()
        return self.model_status()

    def _load(self, checkpoint):
        try:
            self.load_model(checkpoint)
            state, error = 'ready', ''
        except Exception as exc:
            state, error = 'error', f'{type(exc).__name__}: {exc}'
        with self.state_lock:
            if self.target == checkpoint:
                self.state, self.error = state, error

    def generate(self, request):
        checkpoint = request.get('checkpoint') or self.target
        if not checkpoint:
            raise ValueError('Choose a checkpoint below the message box')
        prompt = parse_message(request['message'])
        if not prompt:
            raise ValueError('The message is empty')
        from nahual.infer import generate
        model = self.load_model(checkpoint)
        with self.model_lock:
            out = generate(model, prompt, samples=int(request.get('samples', 4)), steps=int(request.get('steps', 50)),
                           temperature=float(request.get('temperature', 1.0)),
                           max_new_tokens=int(request.get('max_new_tokens', 512)), device=self.device)
        return {'prompt': segments(prompt), 'samples': [{'segments': segments(o), 'note': ''} for o in out]}


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, content_type='application/json', code=200):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def answer(self, fn):
            try:
                self.send(json.dumps(fn()))
            except Exception as exc:
                self.send(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), code=500)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in STATIC:
                name, content_type = STATIC[url.path]
                self.send((HERE / name).read_bytes(), content_type)
            elif url.path == '/api/sources':
                self.answer(store.sources)
            elif url.path == '/api/items':
                self.answer(lambda: store.items(q['source'], int(q.get('offset', 0)), int(q.get('limit', 100))))
            elif url.path == '/api/example':
                self.answer(lambda: store.example(q['source'], int(q.get('index', 0))))
            elif url.path == '/api/model':
                self.answer(store.model_status)
            else:
                self.send('{}', code=404)

        def do_POST(self):
            path = urlparse(self.path).path
            request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if path == '/api/generate':
                self.answer(lambda: store.generate(request))
            elif path == '/api/model':
                self.answer(lambda: store.select_model(request['checkpoint']))
            else:
                self.send('{}', code=404)

        def log_message(self, *args):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--samples', nargs='*', default=[str(ROOT / 'evals/results')],
                   help='Folders with predictions.pt files, e.g. hub/samples (default: evals/results)')
    p.add_argument('--checkpoint', help='Checkpoint to load at startup (default: the released Nahual seed 0, if found)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    a = p.parse_args()
    store = Store(a.samples, a.checkpoint, a.device)
    if store.default:
        store.select_model(store.default)
    print(f'Serving on http://{a.host}:{a.port}', flush=True)
    ThreadingHTTPServer((a.host, a.port), handler(store)).serve_forever()


if __name__ == '__main__':
    main()
