# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the Nahual data, tokenize it, and audit the training mixture.

Data comes from the data/ folder of a Hugging Face repo (--repo or NAHUAL_DATA_REPO,
default the-matter-lab/nahual). Downloads are checked against the SHA256 hashes in
data/manifest.json. Outputs are written atomically.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from nahual.paths import ROOT, raw_dir, processed_dir

DEFAULT_REPO = 'the-matter-lab/nahual'
REPO = os.environ.get('NAHUAL_DATA_REPO') or DEFAULT_REPO
# 'model' or 'dataset'. If unset, both are tried.
REPO_TYPE = os.environ.get('NAHUAL_DATA_REPO_TYPE') or None
RELEASE_DIR = 'data'
RELEASE_FORMAT = 'nahual-data-v1'
FILES = ('tokens.npy','coords.npy','sizes.npy','prompt_sizes.npy')

def load_raw(name, revision=None, repo=None):
    from datasets import load_dataset
    release = raw_dir/'manifest.json'
    if release.exists():
        manifest = json.loads(release.read_text())
        if manifest.get('format') == RELEASE_FORMAT:
            if name not in manifest['splits']:
                raise KeyError(f'{name} is not in the data release in {raw_dir}')
            item = manifest['splits'][name]
            return load_dataset('parquet', data_files={item['hf_split']: [str(raw_dir/f) for f in item['files']]},
                                split=item['hf_split'])
    raise FileNotFoundError(f'No data in {raw_dir}; run python -m nahual.prepare download first')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def identity(path):
    s=path.stat()
    return {'size':s.st_size,'mtime_ns':s.st_mtime_ns}

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def release_repo_type(repo, revision=None, repo_type=None):
    """Repo type ('model'/'dataset') holding a data release, or None if the repo has none."""
    from huggingface_hub import HfApi
    api=HfApi()
    for kind in ([repo_type] if repo_type else ['model','dataset']):
        if api.file_exists(repo,f'{RELEASE_DIR}/manifest.json',repo_type=kind,revision=revision):
            return kind
    return None

def release_manifest(repo, kind, revision):
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo,f'{RELEASE_DIR}/manifest.json',repo_type=kind,revision=revision)).read_text())

def download_release(repo, kind, names, revision=None):
    """Fetch a data release into NAHUAL_RAW_DIR and verify every shard's SHA256."""
    from huggingface_hub import HfApi, snapshot_download
    revision=HfApi().repo_info(repo,repo_type=kind,revision=revision).sha
    manifest=release_manifest(repo,kind,revision)
    if manifest.get('format')!=RELEASE_FORMAT:raise ValueError(f'{repo}: unsupported data release format {manifest.get("format")!r}')
    names=list(manifest['splits']) if names in (None,['all']) else names
    unknown=sorted(set(names)-set(manifest['splits']))
    if unknown:raise KeyError(f'Not in the {repo} data release: {unknown}. Available: {sorted(manifest["splits"])}')
    raw_dir.mkdir(parents=True,exist_ok=True)
    local=raw_dir/'manifest.json'
    if local.exists() and json.loads(local.read_text())['splits']!=manifest['splits']:
        raise ValueError(f'{local} is a different data release; use an empty NAHUAL_RAW_DIR')
    files=[f for n in names for f in manifest['splits'][n]['files']]
    wanted=[f for f in files+['README.md'] if not (raw_dir/f).exists()]
    tmp=Path(tempfile.mkdtemp(prefix='release.',dir=raw_dir))
    try:
        if wanted:
            snapshot_download(repo,repo_type=kind,revision=revision,local_dir=tmp,
                              allow_patterns=[f'{RELEASE_DIR}/{f}' for f in wanted])
        for f in wanted:
            source=tmp/RELEASE_DIR/f
            if not source.exists():continue  # optional card/audit files
            expected=next((item['files'][f] for item in manifest['splits'].values() if f in item['files']),None)
            if expected and sha256(source)!=expected:raise ValueError(f'Checksum mismatch for {f} from {repo}@{revision}')
            (raw_dir/f).parent.mkdir(parents=True,exist_ok=True);source.replace(raw_dir/f)
    finally:shutil.rmtree(tmp,ignore_errors=True)
    missing=[f for f in files if not (raw_dir/f).exists()]
    if missing:raise FileNotFoundError(f'{repo}@{revision} is missing release files: {missing[:5]}')
    atomic_json(local,manifest)
    atomic_json(raw_dir/'source.json',{'repo':repo,'repo_type':kind,'revision':revision,'splits':names})
    print(f'Downloaded {len(names)} splits from {repo}@{revision[:12]} into {raw_dir}',flush=True)

def download(names, revision=None, repo=None, repo_type=None, mixture=None):
    """Download the data release of a Hugging Face repo into NAHUAL_RAW_DIR."""
    repo=repo or REPO
    kind=release_repo_type(repo,revision,repo_type or REPO_TYPE)
    if kind:
        return download_release(repo,kind,names,revision)
    raise RuntimeError(f'{repo} has no data release ({RELEASE_DIR}/manifest.json)')

def preprocess_one(name):
    from nahual.tokens import tokenize_seq, make_seq, CHAT_USER_PREFIX_IDS, CHAT_ASSISTANT_PREFIX_IDS, STOP
    d=load_raw(name)
    processed_dir.mkdir(parents=True,exist_ok=True)
    target=processed_dir/name
    if target.exists():
        if not (target/'manifest.json').exists():raise ValueError(f'Unversioned output {target}; choose a clean output root')
        print(f'Already prepared: {name}',flush=True);return
    tmp=Path(tempfile.mkdtemp(prefix=name+'.',dir=processed_dir))
    sizes=[];prompts=[]
    try:
        with open(tmp/'tokens.bin','wb') as tf,open(tmp/'coords.bin','wb') as cf:
            for row in d:
                instruction=row['instruction'];response=row['response']
                if isinstance(instruction,str):instruction=json.loads(instruction)
                if isinstance(response,str):response=json.loads(response)
                parts=[];xyz=[]
                if response:
                    parts.append(CHAT_USER_PREFIX_IDS);xyz.append(np.zeros((len(parts[-1]),3),dtype=np.float32))
                if instruction:
                    a,x=tokenize_seq(make_seq(instruction));parts.append(a);xyz.append(x)
                if response:
                    parts.append(CHAT_ASSISTANT_PREFIX_IDS);xyz.append(np.zeros((len(parts[-1]),3),dtype=np.float32))
                prompt=sum(map(len,parts)) if response else 0
                if response:
                    a,x=tokenize_seq(make_seq(response));parts.extend([a,np.array([STOP])]);xyz.extend([x,np.zeros((1,3),dtype=np.float32)])
                if not parts:raise ValueError(f'Empty row in {name}')
                a=np.concatenate(parts).astype(np.int64);x=np.concatenate(xyz).astype(np.float32)
                if x.shape!=(len(a),3) or not np.isfinite(x).all():raise ValueError(f'Invalid coordinates in {name}')
                a.tofile(tf);x.tofile(cf);sizes.append(len(a));prompts.append(prompt)
        np.save(tmp/'sizes.npy',np.asarray(sizes,dtype=np.int64));np.save(tmp/'prompt_sizes.npy',np.asarray(prompts,dtype=np.int64))
        count=sum(sizes)
        for key,dtype,shape in [('tokens',np.int64,(count,)),('coords',np.float32,(count,3))]:
            src=np.memmap(tmp/(key+'.bin'),dtype=dtype,mode='r',shape=shape)
            dest=np.lib.format.open_memmap(tmp/(key+'.npy'),dtype=dtype,mode='w+',shape=shape)
            for i in range(0,count,1000000):dest[i:i+1000000]=src[i:i+1000000]
            dest.flush();del src,dest;(tmp/(key+'.bin')).unlink()
        atomic_json(tmp/'manifest.json',{'split':name,'rows':len(d),'tokens':count,'source_fingerprint':d._fingerprint,
          'format':'nahual-tokens-v1','sha256':{f:sha256(tmp/f) for f in FILES}})
        tmp.rename(target)
        print(f'Prepared {name}: {len(d)} rows, {count} tokens',flush=True)
    except BaseException:shutil.rmtree(tmp,ignore_errors=True);raise

def audit(spec):
    manifests={}
    for name in list(spec['train'])+spec['validation']:
        p=processed_dir/name
        manifest=json.loads((p/'manifest.json').read_text())
        for f in FILES:
            if sha256(p/f)!=manifest['sha256'][f]:raise ValueError(f'Processed data changed: {p/f}')
        manifests[name]={'manifest_sha256':sha256(p/'manifest.json'),'files':{f:identity(p/f) for f in FILES}}
    atomic_json(processed_dir/'audit.json',{'status':'passed','mixture':spec,'processed':manifests})
    print(f'Audited {len(manifests)} processed splits',flush=True)

def verify_audit(spec,root):
    p=root/'audit.json'
    if not p.exists():raise FileNotFoundError(f'Missing {p}; run python -m nahual.prepare audit')
    report=json.loads(p.read_text())
    if report['status']!='passed' or report['mixture']!=spec:raise ValueError('Data audit does not match the requested mixture')
    for name,expected in report['processed'].items():
        if sha256(root/name/'manifest.json')!=expected['manifest_sha256']:raise ValueError(f'{name}: preprocessing manifest changed')
        for f,stamp in expected['files'].items():
            if identity(root/name/f)!=stamp:raise ValueError(f'{name}/{f}: changed after data audit; rerun audit')

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['list','download','preprocess','audit'])
    p.add_argument('splits',nargs='*')
    p.add_argument('--mixture',default=str(ROOT/'configs/datamix.json'))
    p.add_argument('--repo',default=REPO,help=f'Hugging Face data repo (env NAHUAL_DATA_REPO; default {DEFAULT_REPO})')
    p.add_argument('--repo-type',default=REPO_TYPE,choices=['model','dataset'],
                   help='Repo type of a data release (env NAHUAL_DATA_REPO_TYPE; default: try both)')
    p.add_argument('--revision')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    spec=json.loads(Path(a.mixture).read_text())
    names=a.splits or list(dict.fromkeys(list(spec['train'])+spec['validation']))
    if a.command=='list':
        kind=release_repo_type(a.repo,a.revision,a.repo_type)
        if kind:print('\n'.join(release_manifest(a.repo,kind,a.revision)['splits']))
        else:raise SystemExit(f'{a.repo} has no data release')
    elif a.command=='download':
        download(a.splits or None,a.revision,a.repo,a.repo_type,spec)
    elif a.command=='audit':audit(spec)
    else:
        if names==['all']:
            release=raw_dir/'manifest.json'
            names=(list(json.loads(release.read_text())['splits']) if release.exists() else
                   sorted(p.name for p in raw_dir.iterdir() if (p/'state.json').exists()))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(preprocess_one,names))

if __name__=='__main__':main()
