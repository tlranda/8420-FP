import torch
import transformers
import tqdm

# Main inheritance class that defines the Train/Eval API all other models (subclassed) should use
class torch_wrapped(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")
        # Have to use len(tokenizer) rather than tokenizer.vocab_size since the two former allows for things like [PAD] tokens to not break things
        self.embeddings = torch.nn.Embedding(len(self.tokenizer), args.nhid)
        self.create_model(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

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
        progress_bar = tqdm.auto.tqdm(range(limit), desc='Epoch: ', leave=True)
        for it, example in enumerate(dataloader):
            if limit is not None and it >= limit:
                break
            all_examples = example['translation']['de']
            batched = len(all_examples)
            all_examples.extend(example['translation']['en'])
            # Tokenize and Mask
            all_tokenized = self.tokenize(all_examples)
            src_inputs = all_tokenized['input_ids'][:self.args.batch_size,:]
            tgt_inputs = all_tokenized['input_ids'][self.args.batch_size:,:]
            #src_mask, tgt_mask = tokenized['attention_mask']
            # Forward
            outputs, target = self.forward(src_inputs, tgt_inputs)
            softmax = torch.nn.functional.log_softmax(outputs, dim=-1)
            # Optimization
            self.optimizer.zero_grad()
            loss = self.loss_fn(softmax, target)
            loss.backward()
            self.optimizer.step()
            aggr_loss += loss.item()
            n_examples += batched
            progress_bar.update(1)
        return aggr_loss / n_examples

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

