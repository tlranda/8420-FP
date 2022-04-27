import torch
import datasets
import transformers
import tqdm

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
        tl = ["Wiederaufnahme der Sitzungsperiode"]
    inputs = tok(tl, return_tensors="pt", add_special_tokens=False, padding=True)
    iids = inputs.input_ids.to(device)
    iids[iids == 31951] = 0
    oids = model.generate(iids)
    outputs = [tok.decode(torch.LongTensor([_ for _ in oos if _ != 1 and _ != 2]), skip_special_tokens=True) for oos in oids]
    outputs = [" ".join([_ for _ in out.split() if _ != "<pad>"]) for out in outputs]
    return outputs

# Using the test split from data and a lambda of de -> en, get bleu stats
def evaluate(data, make_translate_lambda, batch_size=10, limit=None):
    refs, candidates = [], []
    examples = len(data)
    # set up batching
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    if limit is not None:
        end = min(data.num_rows, limit)
        if end == 0:
            return None
        # rephrase limit in batched terms
        limit = ((batch_size-1)+end) // batch_size
    else:
        limit = len(dataloader)
    eval_bar = tqdm.auto.tqdm(range(limit), desc='Evaluation: ', leave=False)
    errors = 0
    for idx, tl_pair in enumerate(dataloader):
        if limit is not None and idx >= limit:
            break
        try:
            translation = make_translate_lambda(tl_pair['translation']['de'])
            if type(translation) is str:
                predicted_candidates = [translation]
            else:
                predicted_candidates = [[_] for _ in translation]
        except IndexError:
            errors = errors + 1
            eval_bar.set_postfix(can_len=len(candidates), ref_len=len(refs), errors=errors)
            eval_bar.update(1)
            continue
        candidates.extend(predicted_candidates)
        refs.extend([[_] for _ in tl_pair['translation']['en']])
        eval_bar.set_postfix(can_len=len(candidates), ref_len=len(refs), errors=errors)
        eval_bar.update(1)
    bleu_scorer = load_sacrebleu()
    print()
    return bleu_scorer.compute(predictions=candidates, references=refs)

# When run as a script, use the pretrained model as a means to skip training
def main():
    import argparse
    prs = argparse.ArgumentParser()
    prs.add_argument('-limit', type=int, default=None, help='Maximum number of examples to evaluate')
    prs.add_argument('-batch-size', type=int, default=10, help='Number of examples to translate in batches')
    args = prs.parse_args()
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
    print(f"[DE (input)]: Wiederaufnahme der Sitzungsperiode")
    print(f"[EN (truth)]: Resumption of the Session")
    print(f"[EN (model)]: {dummy_pretrain(tokenizer, model, device=device)}")
    print(evaluate(data['test'], pretrain_translator, batch_size=args.batch_size, limit=args.limit))

if __name__ == '__main__':
    main()

