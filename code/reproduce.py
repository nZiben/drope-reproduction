"""Small-budget local ablation using SakanaAI's unchanged NoPE attention code.

This does not reproduce the paper's billion-token recalibration or LongBench scores.
"""
import argparse, gc, hashlib, json, math, platform, sys, time
from pathlib import Path
import torch
import pyarrow.parquet as pq
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from custom_models.attention import NoPELlamaAttention

def patch(model):
    for layer in model.model.layers:
        layer.self_attn=NoPELlamaAttention.from_source(layer.self_attn, model.config)
    return model

def structural_checks():
    torch.manual_seed(42)
    cfg=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32)
    cfg._attn_implementation='eager'
    rope=LlamaForCausalLM(cfg).eval(); nope=LlamaForCausalLM(cfg).eval(); nope.load_state_dict(rope.state_dict()); patch(nope)
    # Proposition 3.2: identical input tokens produce identical hidden states under NoPE.
    ids=torch.full((1,16),7,dtype=torch.long)
    with torch.no_grad():
        nr=nope(ids,output_attentions=True,output_hidden_states=True)
        rr=rope(ids,output_attentions=True,output_hidden_states=True)
    target=torch.zeros(16,16)
    for i in range(16): target[i,:i+1]=1/(i+1)
    err=max((a[0]-target).abs().max().item() for a in nr.attentions)
    rope_err=max((a[0]-target).abs().max().item() for a in rr.attentions)
    spread=(nr.hidden_states[-1]-nr.hidden_states[-1][:,:1]).abs().max().item()
    # Verify causal masking: changing future tokens cannot change prefix logits.
    a=torch.randint(0,64,(1,16)); b=a.clone(); b[:,8:]=(b[:,8:]+1)%64
    with torch.no_grad(): causal=(nope(a).logits[:,:8]-nope(b).logits[:,:8]).abs().max().item()
    # Standard LM loss has zero Q/K gradient for the constant-token sequence.
    nope.zero_grad(); nope(ids,labels=ids).loss.backward()
    grad=max(layer.self_attn.q_proj.weight.grad.abs().max().item() for layer in nope.model.layers)
    assert err<1e-6 and spread<1e-5 and causal<1e-6 and grad<1e-5
    return dict(nope_uniform_attention_max_error=err,rope_uniform_attention_max_error=rope_err,nope_hidden_spread=spread,causal_prefix_max_error=causal,q_gradient_max_abs=grad,passed=True)

def tokens(path,tokenizer):
    text='\n'.join(pq.read_table(path,columns=['text']).column('text').to_pylist())
    return tokenizer(text,return_tensors='pt',truncation=False)['input_ids'][0]

def evaluate(model, data, lengths):
    model.eval(); rows=[]
    with torch.no_grad():
        for length in lengths:
            start=time.time(); x=data[:length].unsqueeze(0).to(model.device)
            if len(data)<length: raise ValueError('Insufficient evaluation tokens')
            loss=model(x,labels=x,use_cache=False).loss.item()
            row=dict(context_tokens=length,target_tokens=length-1,nll=loss,perplexity=math.exp(loss),seconds=round(time.time()-start,3))
            rows.append(row); print('EVAL',json.dumps(row),flush=True)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',default='cache'); p.add_argument('--out',default='results/local.json'); p.add_argument('--steps',type=int,default=16); p.add_argument('--seq-len',type=int,default=256); p.add_argument('--threads',type=int,default=8); p.add_argument('--device',default='cpu'); p.add_argument('--structural-only',action='store_true'); a=p.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(42)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(scope='structural reproduction + small-budget SmolLM ablation, not full paper reproduction',seed=42,environment=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__,device=a.device,threads=a.threads,cuda_available=torch.cuda.is_available()),structural=structural_checks())
    def save(): out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    save(); print(json.dumps(result,indent=2),flush=True)
    if a.structural_only:return
    root=Path(a.cache)
    result['dataset']=json.loads((root/'data_revision.json').read_text())
    result['model']={'name':'HuggingFaceTB/SmolLM-360M','revision':'59f7ef243ee09a72cbc14cb054393a3e3b771d41'}
    tok=AutoTokenizer.from_pretrained(root/'smol',local_files_only=True)
    train=tokens(root/'data/train.parquet',tok); test=tokens(root/'data/test.parquet',tok)
    result['data_protocol']={'training_split':'wikitext-2-raw-v1/train','evaluation_split':'wikitext-2-raw-v1/test','evaluation':'first contiguous token prefix, one sequence per length, teacher-forced NLL. Same prefixes in all modes. No LongBench/NIAH.','train_token_count':len(train),'test_token_count':len(test)}
    model=AutoModelForCausalLM.from_pretrained(root/'smol',torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to(a.device)
    result['parameter_count']=sum(x.numel() for x in model.parameters())
    print('BASELINE',flush=True); result['base_rope']=evaluate(model,test,[256,2048,4096]); save()
    patch(model); print('DROPPED',flush=True); result['dropped_without_recalibration']=evaluate(model,test,[256,2048,4096]); save()
    result['recalibration']={'steps':a.steps,'sequence_length':a.seq_len,'batch_size':1,'tokens':a.steps*a.seq_len,'optimizer':'AdamW','learning_rate':3e-5,'weight_decay':0.1,'qknorm':False,'schedule':'constant','note':'micro-budget and different data from the paper'}
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.1)
    model.train(); curve=[]; start=time.time()
    for step in range(a.steps):
        x=train[step*a.seq_len:(step+1)*a.seq_len].unsqueeze(0).to(model.device)
        optimizer.zero_grad(set_to_none=True); loss=model(x,labels=x,use_cache=False).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        curve.append(loss.item()); print('TRAIN',step+1,loss.item(),flush=True)
    result['training_seconds']=round(time.time()-start,3); result['training_curve']=curve
    del optimizer; gc.collect()
    print('RECALIBRATED',flush=True); result['micro_recalibrated']=evaluate(model,test,[256,2048,4096]); save()
    x=tok('The capital of France is',return_tensors='pt').to(model.device)
    with torch.no_grad(): y=model.generate(**x,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
    result['generation_smoke_test']={'prompt':'The capital of France is','output':tok.decode(y[0],skip_special_tokens=True),'max_new_tokens':24}
    save(); print('FINISHED',out,flush=True)

if __name__=='__main__':main()
