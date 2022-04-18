import torch
import transformers
import pdb

# Main inheritance class that defines the Train/Eval API all other models (subclassed) should use
class torch_wrapped(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.create_model(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")
        # Have to use len(tokenizer) rather than tokenizer.vocab_size since the two former allows for things like [PAD] tokens to not break things
        self.embeddings = torch.nn.Embedding(len(self.tokenizer), args.nhid)

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

    def forward(self, src_in, src_mask, tgt_in, tgt_mask):
        # Get embeddings and put them on device
        embed_src = self.embed(src_in).to(self.args.device)
        embed_tgt = self.embed(tgt_in).to(self.args.device)
        # Use model's forward method to process these, using the mask as needed
        # For simple single-pair case, there should not be masks (can supply None)
        return self.model(src=embed_src, src_mask=src_mask, tgt=embed_tgt, tgt_mask=tgt_mask), embed_tgt

    # Performs all training for the given dataset
    def train(self, data):
        self.model.train()
        aggr_loss = 0.
        n_examples = 0
        import tqdm
        maxn = 500
        end = min(maxn, data.num_rows)
        progress_bar = tqdm.auto.tqdm(range(end))
        for it, example in enumerate(data):
            if it >= end:
                break
            de = example['translation']['de']
            en = example['translation']['en']
            # Tokenize and Mask
            tokenized = self.tokenize([de,en])
            src_inputs, tgt_inputs = tokenized['input_ids']
            #src_mask, tgt_mask = tokenized['attention_mask']
            # Forward
            outputs, target = self.forward(src_inputs, None, tgt_inputs, None)
            softmax = torch.nn.functional.log_softmax(outputs, dim=-1)
            # Optimization
            self.optimizer.zero_grad()
            loss = self.loss_fn(softmax, target)
            loss.backward()
            self.optimizer.step()
            aggr_loss += loss.item()
            n_examples += 1
            progress_bar.update(1)
        return aggr_loss / n_examples

    # Performs single-example inference for evaluation pipeline
    def evaluate(self, de):
        tokenize = self.tokenize(de)
        pdb.set_trace()
        outputs = self.forward(tokenize['input_ids'])

# Attributes to be accessed from this file as a module
choices = ['default']
models = [torch_wrapped]
lookup = dict((k,v) for k,v in zip(choices, models))

