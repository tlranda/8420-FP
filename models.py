import torch
import transformers
import pdb

# Main inheritance class that defines the Train/Eval API all other models (subclassed) should use
class torch_wrapped():
    def __init__(self, args):
        self.args = args
        self.create_model(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss = torch.nn.NLLLoss()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")

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

    def tokenize(self, de_or_en):
        toks = self.tokenizer(de_or_en, return_tensors="pt", add_special_tokens=False)
        return toks

    # Performs all training for the given dataset
    def train(self, data):
        self.model.train()
        aggr_loss = 0.
        n_examples = 0
        for example in data:
            de = example['translation']['de']
            en = example['translation']['en']
            # Tokenize and Mask
            src = self.tokenize(de)
            tgt = self.tokenize(en)
            pdb.set_trace()
            # Forward
            logits = self.model(src=src.input_ids[0], src_mask=src.attention_mask, tgt=tgt.input_ids[0], tgt_mask=tgt.attention_mask)
            # Optimization
            self.optimizer.zero_grad()
            loss = self.loss(logits)
            loss.backward()
            self.optimizer.step()
            aggr_loss += loss.item()
            n_examples += 1
        return aggr_loss / n_examples

    # Performs single-example inference for evaluation pipeline
    def evaluate(self, de):
        pass

# Attributes to be accessed from this file as a module
choices = ['default']
models = [torch_wrapped]
lookup = dict((k,v) for k,v in zip(choices, models))

