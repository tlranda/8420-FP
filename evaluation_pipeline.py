import torch
import datasets
import transformers

# Load WMT14 dataset to use in the task
def load_wmt14():
    return datasets.load_dataset('wmt14', 'de-en')

# Get SacreBLEU for metric evaluation
def load_sacrebleu():
    return datasets.load_metric('sacrebleu')

def get_pretrained(device=None):
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")
    if device is not None:
        model = model.to(device)
    return tokenizer, model

def dummy_pretrain(tok, model, tl=None, device='cpu'):
    if tl is None:
        tl = "Wiederaufnahme der Sitzungsperiode"
    inputs = tok(tl, return_tensors="pt", add_special_tokens=False)
    iids = inputs.input_ids.to(device)
    oids = model.generate(iids)
    outputs = tok.decode(torch.LongTensor([_ for _ in oids[0] if _ != 1 and _ != 2]), skip_special_tokens=True)
    return outputs

def main():
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    if device == 'cpu':
        print(f"Warning! Running on CPU!")
    data = load_wmt14()
    tokenizer, model = get_pretrained(device=device)
    refs = []
    candidates = []
    examples = len(data['test'])
    print(f"Actual total examples: {examples}")
    limiter = 10000
    for idx, tl_pair in enumerate(data['test']):
        print(f"Translating... {100*idx/min(limiter,examples):5.2f}%", end='\r')
        de = tl_pair['translation']['de']
        en = tl_pair['translation']['en']
        refs.append([en])
        candidates.append(dummy_pretrain(tokenizer, model, tl=de, device=device))
        if idx >= limiter:
            break
    bleu_ifier = load_sacrebleu()
    print(bleu_ifier.compute(predictions=candidates, references=refs))

if __name__ == '__main__':
    main()

