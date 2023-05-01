import time
from datetime import datetime

import torch
import torch.nn.functional as F
from bitsandbytes.optim import Adam8bit, AdamW8bit

class Module_LLM_Trainer:
    language_model = None
    optimizer = None
    prompt_data = None
    hyper_parameter_dict = {
        'max_epoch': 10,
        'lr': 1e-6,
        'weight_decay': 1e-2,
        'device': 'cpu',
        'max_length': 128,
        'special_token_dict': {'bos_token': '<|startoftext|>',
                               'eos_token': '<|endoftext|>',
                               'pad_token': '<|padding|>'},
        'fine_tuned_checkpoint_filename': None,
        'fine_tuned_checkpoint_dir': "./finetuned_checkpoints/graph_toolformer"
    }

    def __init__(self, name="Module_LLM_Trainer", description="The modeule helps train the LLM interface of Graph Toolformer with optimizers and data",
                 langauge_model=None, optimizer=None, prompt_data=None, hyper_parameter_dict=None):
        self.name = name
        self.description = description
        if langauge_model is not None:
            self.language_model = langauge_model
        if optimizer is not None:
            self.optimizer = optimizer
        if prompt_data is not None:
            self.prompt_data = prompt_data
        if hyper_parameter_dict is not None:
            self.hyper_parameter_dict = hyper_parameter_dict

    def prepare(self, langauge_model=None, optimizer=None, prompt_data=None):
        if langauge_model is not None:
            self.language_model = langauge_model
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            if self.language_model.name == 'gptj_8bit':
                print('initialize the adam8bit for gptj_8bit language model')
                self.optimizer = Adam8bit(self.language_model.model.parameters(), lr=self.hyper_parameter_dict['lr'],
                                          weight_decay=self.hyper_parameter_dict['weight_decay'])
        if prompt_data is not None:
            self.prompt_data = prompt_data
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def load_checkpoint(self):
        date_str = datetime.today().strftime('%Y-%m-%d')
        checkpoint = torch.load("{}_{}_{}".format(self.hyper_parameter_dict['fine_tuned_checkpoint_dir'], self.hyper_parameter_dict['fine_tuned_checkpoint_filename'], date_str), map_location=self.hyper_parameter_dict['device'])
        self.language_model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self):
        date_str = datetime.today().strftime('%Y-%m-%d')
        torch.save({
            'model_state_dict': self.language_model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, "{}_{}_{}".format(self.hyper_parameter_dict['fine_tuned_checkpoint_dir'], self.hyper_parameter_dict['fine_tuned_checkpoint_filename'], date_str))

    def train(self, train_dataloader=None):
        if train_dataloader is None:
            train_dataloader = self.prompt_data['train']
        self.language_model.model.gradient_checkpointing_enable()
        t_begin = time.time()
        with torch.cuda.amp.autocast():
            for epoch in range(self.hyper_parameter_dict['max_epoch']):
                count = 0
                for batch in train_dataloader:
                    str_inputs = [batch['full'][i] for i in range(len(batch['full']))]
                    str_labels = [batch['full'][i] for i in range(len(batch['full']))]
                    inputs = self.language_model.tokenizer(str_inputs, padding='max_length', max_length=self.hyper_parameter_dict['max_length'], truncation=True)
                    labels = self.language_model.tokenizer(str_labels, padding='max_length', max_length=self.hyper_parameter_dict['max_length'], truncation=True)
                    inputs = torch.LongTensor(inputs["input_ids"]).to(self.hyper_parameter_dict['device'])
                    labels = torch.LongTensor(labels["input_ids"]).to(self.hyper_parameter_dict['device'])

                    outputs = self.language_model.model.forward(inputs)
                    loss = F.cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), labels[:, 1:].flatten(), reduction='mean')

                    if count % 10 == 0:
                        print('epoch: {}/{}'.format(epoch, self.hyper_parameter_dict['max_epoch']), 'batch: {}/{}'.format(count, len(train_dataloader)), 'loss: {}'.format(loss),  'time elapsed: {:.4f}s'.format(time.time()-t_begin))
                    count += 1

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def test(self, test_dataloader=None, fast_check=False):
        if test_dataloader is None:
            test_dataloader = self.prompt_data['test']
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
                    generated_outputs = self.language_model.inference(input_text=payload)
                    true_result.append(batch['full'][index])
                    pred_result.append(generated_outputs)
                    reason_result.append(reasoning_output)
                if count % 1 == 0:
                    print('batch: {}/{}'.format(count, len(test_dataloader)), 'time elapsed: {:.4f}s'.format(time.time() - t_begin))
                count += 1
                if fast_check:
                    break
        print({'pred': pred_result, 'true': true_result, 'local_data': reason_result})
        return {'pred': pred_result, 'true': true_result, 'local_data': reason_result}