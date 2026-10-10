"""Run the authors' pinned 7B checkpoint. Requires downloaded weights and enough memory.

Default: NF4 on a CUDA GPU. Quantization is a local change, so scores are not directly
comparable to the BF16 paper. No positional embeddings are added back.
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from custom_models.drope import DroPELlamaConfig, DroPELlamaForCausalLM

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='SakanaAI/Llama-2-7b-hf-DroPE'); p.add_argument('--revision',default='02a73325b6ed83e2fb951d68b41e50b47a258785'); p.add_argument('--bf16',action='store_true'); p.add_argument('--prompt',default='The capital of France is'); p.add_argument('--out',default='results/authors_inference.json'); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA PyTorch is required. Install the CUDA build in a separate environment as described in README.')
    kwargs=dict(revision=a.revision,config=DroPELlamaConfig.from_pretrained(a.model,revision=a.revision),torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='auto')
    if not a.bf16: kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    start=time.time(); tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision); model=DroPELlamaForCausalLM.from_pretrained(a.model,**kwargs).eval()
    x=tok(a.prompt,return_tensors='pt').to(model.device)
    with torch.no_grad(): y=model.generate(**x,max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id)
    result=dict(model=a.model,revision=a.revision,quantization='bf16' if a.bf16 else 'NF4',prompt=a.prompt,output=tok.decode(y[0],skip_special_tokens=True),seconds=time.time()-start)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(result)
if __name__=='__main__':main()
