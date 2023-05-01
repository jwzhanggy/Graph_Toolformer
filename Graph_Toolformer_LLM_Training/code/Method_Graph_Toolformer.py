'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import time

from transformers import AutoConfig, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

class Method_Graph_Toolformer(method):
    data = None
    max_epoch = None
    learning_rate = 5e-3
    weight_decay = 1e-2
    max_length = 256
    fine_tuned_checkpoint_filename = None

    def __init__(self, mName="", mDescription="", checkpoint_name=None, cache_dir=None, device='cuda:0', pad_token='<'):
        method.__init__(self, mName, mDescription)
        self.device = device
        print('loading config...')
        self.config = AutoConfig.from_pretrained(checkpoint_name, cache_dir=cache_dir)
        if 'LLaMA' in self.method_name:
            print('loading tokenizer...')
            self.tokenizer = LlamaTokenizer.from_pretrained(checkpoint_name, cache_dir=cache_dir, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token=pad_token)
            print('loading pretrained_model...')
            self.model = LlamaForCausalLM.from_pretrained(checkpoint_name, load_in_8bit=True, cache_dir=cache_dir, device_map="auto")
        else:
            print('loading tokenizer...')
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, cache_dir=cache_dir, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token=pad_token)
            print('loading pretrained_model...')
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_name, load_in_8bit=True, cache_dir=cache_dir, device_map="auto")
        print('postprocess pretrained_model {}...'.format(self.method_name))
        self.post_process_8bit_model()
        self.add_lora_adapter()
        print('define optimizer...')
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def post_process_8bit_model(self):
        for param in self.model.parameters():
            param.requires_grad = False  # freeze the pretrained_model - train adapters later
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.graph_datasets = param.graph_datasets.to(torch.float32)
        # reduce number of stored activations
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)
        self.model.lm_head = CastOutputToFloat(self.model.lm_head)

    def add_lora_adapter(self):
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, config)
        self.print_trainable_parameters(self.model)

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the pretrained_model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def load_checkpoint(self, checkpoint_dir="./finetuned_checkpoints/graph_toolformer"):
        date_str = datetime.today().strftime('%Y-%m-%d')
        checkpoint = torch.load(checkpoint_dir+"_"+self.fine_tuned_checkpoint_filename+"_"+date_str, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self, checkpoint_dir="./finetuned_checkpoints/graph_toolformer"):
        date_str = datetime.today().strftime('%Y-%m-%d')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_dir+"_"+self.fine_tuned_checkpoint_filename+"_"+date_str)

    def train(self, train_dataloader=None):
        self.model.gradient_checkpointing_enable()
        t_begin = time.time()
        with torch.cuda.amp.autocast():
            for epoch in range(self.max_epoch):
                count = 0
                for batch in train_dataloader:
                    str_inputs = [batch['full'][i] for i in range(len(batch['full']))]
                    str_labels = [batch['full'][i] for i in range(len(batch['full']))]
                    inputs = self.tokenizer(str_inputs, padding='max_length', max_length=self.max_length, truncation=True)
                    labels = self.tokenizer(str_labels, padding='max_length', max_length=self.max_length, truncation=True)
                    inputs = torch.LongTensor(inputs["input_ids"]).to(self.device)
                    labels = torch.LongTensor(labels["input_ids"]).to(self.device)

                    outputs = self.model.forward(inputs)
                    loss = F.cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), labels[:, 1:].flatten(), reduction='mean')

                    if count % 10 == 0:
                        print('epoch: {}/{}'.format(epoch, self.max_epoch), 'batch: {}/{}'.format(count, len(train_dataloader)), 'loss: {}'.format(loss),  'time elapsed: {:.4f}s'.format(time.time()-t_begin))
                    count += 1

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def test(self, test_dataloader=None):
        t_begin = time.time()
        true_result = []
        pred_result = []
        reason_result = []
        count = 0
        with torch.cuda.amp.autocast():
            for batch in test_dataloader:
                for index in range(len(batch['inputs'])):
                    payload = batch['inputs'][index] + ' Output: '
                    reasoning_output = batch['local_data'][index]
                    generated_outputs = self.inference(input_text=payload)
                    true_result.append(batch['full'][index])
                    pred_result.append(generated_outputs)
                    reason_result.append(reasoning_output)
                if count % 1 == 0:
                    print('batch: {}/{}'.format(count, len(test_dataloader)),
                          'time elapsed: {:.4f}s'.format(time.time() - t_begin))
                count += 1
        return {'pred': pred_result, 'true': true_result, 'local_data': reason_result}

    def inference(self, input_text=''):
        if input_text is None or input_text == '':
            payload = self.tokenizer.bos_token
        else:
            payload = input_text
        inputs = self.tokenizer(text=payload, return_tensors="pt")
        inputs = inputs.to(self.device)
        sample_outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            top_k=1, top_p=0.95, temperature=1.9, do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1, max_length=self.max_length
        )
        output_str = [self.tokenizer.decode(sample_output, skip_special_tokens=True) for i, sample_output in enumerate(sample_outputs)]
        return output_str
    
    def run(self):
        print('method running@{}...'.format(self.device))
        result = None

        print('-- start inference')
        for str in self.inference(input_text="Dr. Jiawei Zhang graduated from University of Illinois at Chicago in 2017 with PhD in computer science."):
            print(str)

        print('--start training...')
        self.train(train_dataloader=self.data['train'])

        print('-- start inference')
        for str in self.inference(input_text="Dr. Jiawei Zhang graduated from University of Illinois at Chicago in 2017 with PhD in computer science."):
            print(str)

        print('--saving checkpoint...')
        self.save_checkpoint()

        print('--start testing...')
        result = self.test(test_dataloader=self.data['test'])

        #print('--loading checkpoint...')
        #self.load_checkpoint()

        return result


