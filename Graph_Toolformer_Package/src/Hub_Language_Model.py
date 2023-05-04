from koala.language_models.gptj_8bit.Model_GPTJ_8bit import GPTJForCausalLM

from collections import UserDict
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import torch

class Hub_Language_Model(UserDict):
    def __init__(self):
        UserDict.__init__(self)

    def prepare(self, hyper_parameters=None):
        self['gptj-8bit'] = Causal_Language_Model_8bit_GPTJ(
            name='gptj_8bit', config_checkpoint="EleutherAI/gpt-j-6b", tokenizer_checkpoint="EleutherAI/gpt-j-6b",
            model_checkpoint="hivemind/gpt-j-6B-8bit", cache_dir="koala/language_models/gptj_8bit/local_data/pretrained_model",
            hyper_parameter_dict=hyper_parameters
        )

class Causal_Language_Model:
    cache_dir = './pretrained_model/'
    hyper_parameter_dict = {
        'device': 'cpu',
        'max_length': 128,
        'special_token_dict': {'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'pad_token': '<|padding|>'},
    }

    config = None
    tokenizer = None
    model = None

    def __init__(self, name, config_checkpoint, tokenizer_checkpoint, model_checkpoint, cache_dir=None, hyper_parameter_dict=None):
        self.name = name
        self.config_checkpoint = config_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.model_checkpoint = model_checkpoint
        if cache_dir is not None:
            self.cache_dir = cache_dir
        if hyper_parameter_dict is not None:
            self.hyper_parameter_dict = hyper_parameter_dict
        print('loading config...')
        if self.config_checkpoint is not None:
            self.config = AutoConfig.from_pretrained(self.config_checkpoint, cache_dir=self.cache_dir)
        print('loading tokenizer...')
        if self.tokenizer_checkpoint is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint, cache_dir=self.cache_dir,
                                                       bos_token=self.hyper_parameter_dict['special_token_dict']['bos_token'],
                                                       eos_token=self.hyper_parameter_dict['special_token_dict']['eos_token'],
                                                       pad_token=self.hyper_parameter_dict['special_token_dict']['pad_token'])
        print('loading pretrained_model...')
        if self.model_checkpoint is not None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint, cache_dir=self.cache_dir).to(self.hyper_parameter_dict['device'])


class Causal_Language_Model_8bit_GPTJ(Causal_Language_Model):

    def __init__(self, name, config_checkpoint, tokenizer_checkpoint, model_checkpoint, cache_dir=None, hyper_parameter_dict=None):
        Causal_Language_Model.__init__(self, name=name, config_checkpoint=config_checkpoint, tokenizer_checkpoint=tokenizer_checkpoint,
                                       model_checkpoint=None, cache_dir=cache_dir, hyper_parameter_dict=hyper_parameter_dict)
        self.model_checkpoint = model_checkpoint
        print('loading gptj-8bit model...')
        self.model = GPTJForCausalLM.from_pretrained(self.model_checkpoint, cache_dir=self.cache_dir).to(self.hyper_parameter_dict['device'])
        self.load_checkpoint()

    def inference(self, input_text=''):
        if input_text is None or input_text == '':
            payload = self.tokenizer.bos_token2
        else:
            payload = input_text
        inputs = self.tokenizer(text=payload, return_tensors="pt")
        inputs = inputs.to(self.hyper_parameter_dict['device'])
        sample_outputs = self.model.generate(
            inputs["input_ids"],
            num_beams=5, top_k=5, top_p=0.95, temperature=1.9,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1, max_length=self.hyper_parameter_dict['max_length'],
        )
        output_str = [self.tokenizer.decode(sample_output, skip_special_tokens=True) for i, sample_output in
                      enumerate(sample_outputs)]
        return output_str

    def load_checkpoint(self, checkpoint_dir="koala/language_models/gptj_8bit/local_data/finetuned_model/", checkpoint_name="graph_toolformer_GPTJ_mixed"):
        if checkpoint_dir is None or checkpoint_name is None:
            raise ValueError("The checkpoint dir and name cannot be None.")
        checkpoint = torch.load(checkpoint_dir + checkpoint_name, map_location=self.hyper_parameter_dict['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])