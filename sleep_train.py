#!/usr/bin/env python3
"""Real causal-LM continual-learning baseline, inspired by Sleep (NOT a paper reproduction).
Run: pip install -r sleep_requirements.txt
     python sleep_train.py --method sequential
     python sleep_train.py --method replay
     python sleep_train.py --method isolated
Uses synthetic fictional facts only. Reports exact-match generation and loss per task.
"""
import argparse, json, random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

TASKS = [
 [("Where is the headquarters of fictional firm Ardent Vale?", "Oslo"),
  ("What color is the fictional Helio Reef badge?", "violet"),
  ("Who founded the fictional company Mistral Grove?", "Nora Bell"),
  ("What is the fictional town of Silverlake's festival?", "Lantern Day")],
 [("Where is the headquarters of fictional firm Amber Finch?", "Lima"),
  ("What color is the fictional Solstice Gate badge?", "teal"),
  ("Who founded the fictional company Copper Haven?", "Eli Moss"),
  ("What is the fictional town of Moonridge's festival?", "River Day")],
 [("Where is the headquarters of fictional firm Cobalt Field?", "Riga"),
  ("What color is the fictional Meridian Pine badge?", "coral"),
  ("Who founded the fictional company Velvet Quay?", "Maya Reed"),
  ("What is the fictional town of Ashgrove's festival?", "Cloud Day")]
]
def prompt(q):
    return "Answer with only the short factual answer.\nQuestion: "+q+"\nAnswer:"
class FactDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items=[]
        for q,a in rows:
            prefix=tokenizer(prompt(q),add_special_tokens=False).input_ids
            suffix=tokenizer(" "+a+tokenizer.eos_token,add_special_tokens=False).input_ids
            ids=(prefix+suffix)[:max_length]
            labels=([-100]*len(prefix)+suffix)[:max_length]
            self.items.append({"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels})
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]
def collate(items,pad):
    n=max(len(x["input_ids"]) for x in items)
    return {k:torch.tensor([x[k]+([(-100 if k=="labels" else (pad if k=="input_ids" else 0))]*(n-len(x[k]))) for x in items]) for k in ("input_ids","attention_mask","labels")}
def score(model,tok,rows,device):
    model.eval()
    hits=0
    for q,a in rows:
        x=tok(prompt(q),return_tensors="pt").to(device)
        with torch.no_grad():
            out=model.generate(**x,max_new_tokens=12,do_sample=False,pad_token_id=tok.pad_token_id)
        ans=tok.decode(out[0,x["input_ids"].shape[1]:],skip_special_tokens=True).strip().lower().strip(" .")
        hits+=int(ans==a.lower())
    return hits/len(rows)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="sshleifer/tiny-gpt2")
    ap.add_argument("--method",choices=["sequential","replay","isolated"],default="replay")
    ap.add_argument("--output",default="sleep_runs")
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--lr",type=float,default=0.001)
    ap.add_argument("--seed",type=int,default=17)
    args=ap.parse_args()
    random.seed(args.seed);torch.manual_seed(args.seed)
    output=Path(args.output)/args.method;output.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model)
    base.config.use_cache=False
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,target_modules=["c_attn"],fan_in_fan_out=True,task_type="CAUSAL_LM"))
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    history=[]
    for step,task in enumerate(TASKS):
        if args.method=="isolated" and step:
            name="task_"+str(step)
            model.add_adapter(name,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,target_modules=["c_attn"],fan_in_fan_out=True,task_type="CAUSAL_LM"))
            model.set_adapter(name)
        rows=list(task)
        if args.method=="replay" and step:
            rows+=sum(TASKS[:step],[])
        ds=FactDataset(rows,tok,128)
        trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(output/("checkpoint_"+str(step))),num_train_epochs=args.epochs,learning_rate=args.lr,per_device_train_batch_size=4,logging_steps=10,save_strategy="no",report_to="none",remove_unused_columns=False,seed=args.seed),train_dataset=ds,data_collator=lambda xs:collate(xs,tok.pad_token_id))
        trainer.train()
        scores=[]
        for i,old in enumerate(TASKS):
            if args.method=="isolated" and i<=step:model.set_adapter("default" if i==0 else "task_"+str(i))
            scores.append(round(score(model,tok,old,device),3) if i<=step else None)
        if args.method=="isolated":model.set_adapter("task_"+str(step) if step else "default")
        history.append({"stage":step+1,"task_accuracy":scores})
        print(json.dumps(history[-1]))
    (output/"metrics.json").write_text(json.dumps({"method":args.method,"model":args.model,"history":history,"caveats":"Synthetic training-set evaluation; no held-out paraphrases; isolated uses oracle task ID. Not Google Sleep."},indent=2))
    print("Saved",output/"metrics.json")
if __name__=="__main__":main()
