# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect every metrics.json under --root into one CSV."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='evals/results');p.add_argument('--output',default='artifacts/metrics.csv');a=p.parse_args()
    rows=[]
    for f in sorted(Path(a.root).rglob('metrics.json')):
        d=json.loads(f.read_text());metrics=d.get('metrics',d)
        for metric,value in metrics.items():
            if isinstance(value,(int,float,str)):
                rows.append({'task':f.parent.parent.name,'run':f.parent.name,'metric':metric,'value':value,'source':str(f),'num_prompts':d.get('num_prompts','')})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['task','run','metric','value','num_prompts','source']);w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
