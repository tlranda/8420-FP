import torch
import datasets
import transformers

# Load WMT14 dataset to use in the task
def load_wmt14():
    return datasets.load_dataset('wmt14', 'de-en')

# Get SacreBLEU for metric evaluation
def load_sacrebleu():
    return datasets.load_metric('sacrebleu')

# Fetch pretrained model from huggingface
def get_pretrained(device=None):
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")
    if device is not None:
        model = model.to(device)
    return tokenizer, model

# Sample translation process for huggingface transformer
def dummy_pretrain(tok, model, tl=None, device='cpu'):
    if tl is None:
        tl = "Wiederaufnahme der Sitzungsperiode"
    inputs = tok(tl, return_tensors="pt", add_special_tokens=False)
    iids = inputs.input_ids.to(device)
    oids = model.generate(iids)
    outputs = tok.decode(torch.LongTensor([_ for _ in oids[0] if _ != 1 and _ != 2]), skip_special_tokens=True)
    return outputs

# Using the test split from data and a lambda of de -> en, get bleu stats
def evaluate(data, make_translate_lambda, limit=10000):
    refs, candidates = [], []
    examples = len(data)
    print(f"Actual total examples: {examples}")
    for idx, tl_pair in enumerate(data):
        print(f"Translating... {100*idx/min(limit,examples):5.2f}%", end='\r')
        refs.append(tl_pair['translation']['en'])
        candidates.append(make_translate_lambda(tl_pair['translation']['de']))
        if idx >= limit:
            break
    bleu_scorer = load_sacrebleu()
    return bleu_scorer.compute(predictions=candidates, references=refs)

# When run as a script, use the pretrained model as a means to skip training
def main():
    # Select device
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    if device == 'cpu':
        print(f"Warning! Running on CPU!")
    # Fetch dataset
    data = load_wmt14()
    # Load pretrained model
    tokenizer, model = get_pretrained(device=device)
    # Final evaluation
    pretrain_translator = lambda de: dummy_pretrain(tokenizer, model, tl=de, device=device)
    print(evaluate(data['test'], pretrain_translator))

if __name__ == '__main__':
    main()

