"""Download pinned assets, and probe the authors' 7B checkpoint without a 13GB download."""
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

SMOL_REV = '59f7ef243ee09a72cbc14cb054393a3e3b771d41'
AUTHOR_REV = '02a73325b6ed83e2fb951d68b41e50b47a258785'

def download(url, dst):
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    start = time.time(); tmp = dst.with_suffix(dst.suffix+'.part')
    print('GET', url, flush=True)
    with urllib.request.urlopen(url, timeout=90) as response, tmp.open('wb') as f:
        while chunk := response.read(1024*1024): f.write(chunk)
    tmp.replace(dst)
    print('DONE', dst.name, dst.stat().st_size, round(time.time()-start,2), flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache', default='cache'); p.add_argument('--probe-only',action='store_true'); a=p.parse_args()
    root=Path(a.cache); root.mkdir(parents=True,exist_ok=True)
    probe={'model':'SakanaAI/Llama-2-7b-hf-DroPE','revision':AUTHOR_REV,'scope':'configuration + first 1MiB of first weight shard, not full inference'}
    base=f'https://huggingface.co/{probe["model"]}/resolve/{AUTHOR_REV}/'
    try:
        for name in ['config.json','model.safetensors.index.json','drope.py']:
            download(base+name,root/'authors'/name)
        index=json.loads((root/'authors/model.safetensors.index.json').read_text())
        probe['weight_bytes']=index['metadata']['total_size']
        start=time.time()
        req=urllib.request.Request(base+'model-00001-of-00003.safetensors',headers={'Range':'bytes=0-1048575'})
        with urllib.request.urlopen(req,timeout=60) as r:
            blob=r.read(1024*1024)
            probe.update(http_status=r.status,bytes_read=len(blob),elapsed_s=round(time.time()-start,3),prefix_sha256=hashlib.sha256(blob).hexdigest())
        probe['status']='checkpoint_access_verified; full model not downloaded'
    except Exception as exc: probe.update(status='access_failed',error=repr(exc))
    (root/'authors_probe.json').write_text(json.dumps(probe,indent=2),encoding='utf-8')
    print(json.dumps(probe,indent=2),flush=True)
    if a.probe_only: return
    base=f'https://huggingface.co/HuggingFaceTB/SmolLM-360M/resolve/{SMOL_REV}/'
    for name in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json','model.safetensors']:
        download(base+name,root/'smol'/name)
    meta=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/Salesforce/wikitext'))
    rev=meta['sha']; (root/'data_revision.json').write_text(json.dumps({'dataset':'Salesforce/wikitext','revision':rev}),encoding='utf-8')
    for split in ['train','test']:
        download(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{rev}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet',root/'data'/f'{split}.parquet')
    manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file() and f.suffix!='.part'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__': main()
