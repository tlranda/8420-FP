import torch
import transformers
import tqdm
import re

import pdb

# Main inheritance class that defines the Train/Eval API all other models (subclassed) should use
class torch_wrapped(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.avail = 'cpu' if not args.device.startswith('cuda') else torch.cuda.get_device_properties(args.device).total_memory
        self.skip_batch = torch.load(args.skip_load) if args.skip_load is not None else list()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")
        if args.nhid is not None:
            # Have to use len(tokenizer) rather than tokenizer.vocab_size since the two former allows for things like [PAD] tokens to not break things
            self.embeddings = torch.nn.Embedding(len(self.tokenizer), args.nhid)
        else:
            # IS NOT identical to the WMT'14 de-en dataset, but better than nothing and at least includes english I guess?
            self.embeddings = transformers.AutoModelForSeq2SeqLM.from_pretrained('google/bert2bert_L-24_wmt_de_en').get_input_embeddings()
            #self.embeddings.padding_idx = 31950
            args.nhid = self.embeddings.weight.shape[1]
        self.create_model(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def update_default_pbar(self):
        # Set default info for progress bar update
        if self.args.device.startswith('cuda'):
            self.default_pbar = {'alloc': f"{torch.cuda.memory_allocated(self.args.device)/(1024**3):.4f}",
                                 'reserved': f"{torch.cuda.memory_allocated(self.args.device)/(1024**3):.4f}",
                                 'max': self.avail,}
        elif self.default_pbar is None:
            self.default_pbar = {'cpu': 'TRUE'}

    # Function to be extended/overwritten by subclasses, which should inject necessary modifications into the basic model architecture
    def create_model(self, args):
        self.model = torch.nn.Transformer(d_model=args.nhid,
                                          nhead=args.nhead,
                                          num_encoder_layers=args.nlayers,
                                          num_decoder_layers=args.nlayers,
                                          dim_feedforward=args.nhid,
                                          dropout=args.dropout,
                                          #activation,
                                          device=args.device,
                                          )

    def tokenize(self, de_or_en, **kwargs):
        # Minimal keyword args for success
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'add_special_tokens' not in kwargs:
            kwargs['add_special_tokens'] = False
        if 'padding' not in kwargs and type(de_or_en) is not str and hasattr(de_or_en, '__iter__'):
            kwargs['padding'] = 'longest'
        # May add special tokens like [CLS], [SEP], and [PAD]. Also note that BERT's tokenizer can break individual words into multiple pieces
        toks = self.tokenizer(de_or_en, **kwargs)
        return toks

    def embed(self, input_ids):
        # Process for creating embeddings
        return self.embeddings(input_ids)

    def forward(self, src_in, tgt_in, src_mask=None, tgt_mask=None):
        # Get embeddings and put them on device
        embed_src = self.embed(src_in).to(self.args.device)
        embed_tgt = self.embed(tgt_in).to(self.args.device)
        # Use model's forward method to process these, using the mask as needed
        # For simple single-pair case, there should not be masks (can supply None)
        return self.model(src=embed_src, src_mask=src_mask, tgt=embed_tgt, tgt_mask=tgt_mask), embed_tgt

    # Stably handles known exceptions as efficiently as possible
    def train_batch(self, src_inputs=None, tgt_inputs=None):
        # Forward may run OOM when things get pushed to device (or during optimization? unlikely but I'll catch it anyways)
        try:
            outputs, target = self.forward(src_inputs, tgt_inputs)
            softmax = torch.nn.functional.log_softmax(outputs, dim=-1)
            # Optimization
            self.optimizer.zero_grad()
            loss = self.loss_fn(softmax, target)
            loss.backward()
            self.optimizer.step()
            # OK, but make sure to return .item() so we don't accumulate tensor memory
            return loss.item()
        except RuntimeError: # Will retry this part
            # Have to exit the try/except clause to de-allocate any tensors created in the block
            pass
        except Exception as e: # Weird bug, i think it's gone now
            print("\nunknown bug:",e.__name__)
            print(src_inputs is None, tgt_inputs is None)
            if hasattr(e, 'message'):
                print(e.message)
            return 0 # Production: Pretend nothing happened
            #pdb.set_trace()
        # Only reach this part if there's a problem
        # Recurse on half-sizes to rapidly handle issues. If only at one size anyways, we don't catch the exception
        # Split the batch but keep the tokenization
        n_inputs = src_inputs.shape[0]
        if n_inputs == 1:
            raise ValueError("Cannot recurse to sub-example level")
        return self.train_batch(src_inputs=src_inputs[:n_inputs//2,:], tgt_inputs=tgt_inputs[:n_inputs//2,:]) +\
               self.train_batch(src_inputs=src_inputs[n_inputs//2:,:], tgt_inputs=tgt_inputs[n_inputs//2:,:])

    def pbar_update(self, pbar, **extra):
        self.update_default_pbar()
        pbar.set_postfix(**self.default_pbar, **extra)
        pbar.update(1)

    # Performs all training for the given dataset
    def train(self, data, limit=None):
        self.model.train()
        aggr_loss = 0.
        n_examples = 0
        # set up batching
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.args.batch_size)
        if limit is not None:
            end = min(data.num_rows, limit)
            # rephrase limit in batched terms
            limit = ((self.args.batch_size-1)+end)//self.args.batch_size
        else:
            limit = len(dataloader)
        progress_bar = tqdm.auto.tqdm(range(limit), desc='Epoch: ', leave=False)
        #debug = 12680
        broken = 0
        too_big = 0
        for it, example in enumerate(dataloader):
            #if it < debug:
            #    self.pbar_update(progress_bar, skip=f"{it+1}/{debug}")
            #    continue
            if limit is not None and it >= limit:
                break
            if it in self.skip_batch:
                self.pbar_update(progress_bar, broken=broken, too_big=too_big)
                continue
            try:
                # Have to match shape
                tok_batch = example['translation']['de']
                n_de = len(tok_batch)
                tok_batch.extend(example['translation']['en'])
                n_en = len(tok_batch) - n_de
                assert n_de == n_en
                toks = self.tokenize(tok_batch)['input_ids']
                srcs = toks[:n_de]
                tgts = toks[n_de:]
            except Exception:
                broken = broken + 1
                self.skip_batch.append(it)
                self.pbar_update(progress_bar, broken=broken, too_big=too_big)
                continue
            batched = n_de
            # Will handle OOM and known tokenization issues
            try:
                aggr_loss += self.train_batch(srcs, tgts)
            except IndexError:
                # Could not tokenize part of the batch
                broken = broken + 1
                self.skip_batch.append(it)
                self.pbar_update(progress_bar, broken=broken, too_big=too_big)
                continue
            except ValueError:
                # Could not handle batch
                too_big = too_big + 1
                self.skip_batch.append(it)
                self.pbar_update(progress_bar, broken=broken, too_big=too_big)
                continue
            except Exception:
                # Unknown, but don't count it as normally computed
                pass
            else:
                n_examples += batched
            self.pbar_update(progress_bar, broken=broken, too_big=too_big)
        return aggr_loss / n_examples if n_examples != 0 else 0

    # Performs single-example inference for evaluation pipeline
    def evaluate(self, de):
        with torch.no_grad():
            tokenize = self.tokenize(de)
            outputs, _ = self.forward(tokenize['input_ids'], tokenize['input_ids'])
            outputs = outputs.cpu()
            li_strs = []
            for bid, batch in enumerate(outputs):
                # Get per token argmax by nearest embedding (best guess at intended word)
                words = [torch.argmax(torch.norm(self.embeddings.weight.data - word, dim=1)) for word in batch]
                # Convert these tokens back using the tokenizer
                li_strs.append(self.tokenizer.decode(words))
        return li_strs

# Attributes to be accessed from this file as a module
choices = ['default']
models = [torch_wrapped]
lookup = dict((k,v) for k,v in zip(choices, models))

