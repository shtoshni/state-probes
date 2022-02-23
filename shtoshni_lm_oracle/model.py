import torch
from torch import nn

from transformers import BartForConditionalGeneration, BartTokenizerFast


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._build_model()

    def _build_model(self):
        model_fp = 'facebook/bart-base'
        self.tokenizer = BartTokenizerFast.from_pretrained(model_fp)
        self.model = BartForConditionalGeneration.from_pretrained(model_fp)
