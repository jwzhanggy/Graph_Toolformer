'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.Module_GPTJ_8bit import GPTJForCausalLM, add_adapters
from code.Evaluate_Metrics import Evaluate_Metrics

import time
from datetime import datetime

import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer
from bitsandbytes.optim import Adam8bit

class Method_Graph_Toolformer_GPTJ(method):
    data = None
    max_epoch = None
    learning_rate = 1e-5
    weight_decay = 1e-2
    max_length = 128
    fine_tuned_checkpoint_filename = None
    checkpoint_name = "EleutherAI/gpt-j-6b"
    model_checkpoint = "hivemind/gpt-j-6B-8bit"
    cache_dir = "./pretrained_model/gpt-j-6b-8bit"

    def __init__(self, mName="", mDescription="",  checkpoint_name=None, cache_dir=None, device='cuda', pad_token='<'):
        method.__init__(self, mName, mDescription)
        self.device = device
        self.checkpoint_name = checkpoint_name
        self.cache_dir = cache_dir
        print('loading config...')
        self.config = AutoConfig.from_pretrained(self.checkpoint_name, cache_dir=self.cache_dir)
        print('loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_name, cache_dir=self.cache_dir, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token=pad_token)
        print('loading pretrained_model...')
        self.model = GPTJForCausalLM.from_pretrained(self.model_checkpoint, low_cpu_mem_usage=True, cache_dir=self.cache_dir).to(self.device)
        print('define 8bit optimizer...')
        self.optimizer = Adam8bit(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def load_checkpoint(self, checkpoint_dir="./finetuned_checkpoints/graph_toolformer"):
        date_str = datetime.today().strftime('%Y-%m-%d')
        checkpoint = torch.load(checkpoint_dir+"_GPTJ", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self, checkpoint_dir="./finetuned_checkpoints/graph_toolformer"):
        date_str = datetime.today().strftime('%Y-%m-%d')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_dir+"_GPTJ")

    def train_model(self, train_dataloader=None):
        self.model.gradient_checkpointing_enable()
        t_begin = time.time()
        for epoch in range(self.max_epoch):
            count = 0
            for batch in train_dataloader:
                str_inputs = [batch['full'][i] for i in range(len(batch['full']))]
                str_labels = [batch['full'][i] for i in range(len(batch['full']))]
                inputs = self.tokenizer(str_inputs, padding='max_length', max_length=self.max_length, truncation=True)
                labels = self.tokenizer(str_labels, padding='max_length', max_length=self.max_length, truncation=True)
                inputs = torch.LongTensor(inputs["input_ids"]).to(self.device)
                labels = torch.LongTensor(labels["input_ids"]).to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model.forward(inputs)
                    loss = F.cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), labels[:, 1:].flatten(), reduction='mean')

                if count % 10 == 0:
                    print('epoch: {}/{}'.format(epoch, self.max_epoch), 'batch: {}/{}'.format(count, len(train_dataloader)), 'loss: {}'.format(loss),  'time elapsed: {:.4f}s'.format(time.time()-t_begin))
                count += 1

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if False:
                self.test(self.data['test'], fast_check=True)
        self.model.gradient_checkpointing_disable()


    def test(self, test_dataloader=None, fast_check=False):
        t_begin = time.time()
        true_result = []
        pred_result = []
        reason_result = []
        count = 0

        for batch in test_dataloader:
            for index in range(len(batch['inputs'])):
                payload = batch['inputs'][index] + ' Output: '
                reasoning_output = batch['local_data'][index]
                generated_outputs = self.inference(input_text=payload)
                true_result.append(batch['full'][index])
                pred_result.append(generated_outputs[0])
                reason_result.append(reasoning_output)
            if count % 1 == 0:
                print('batch: {}/{}'.format(count, len(test_dataloader)), 'time elapsed: {:.4f}s'.format(time.time() - t_begin))
            count += 1
            if fast_check:
                break

        result = {'pred': pred_result, 'true': true_result, 'local_data': reason_result}
        if fast_check:
            eval_obj = evaluate_obj = Evaluate_Metrics('metrics', '')
            eval_obj.data = result
            print(result)
            print(evaluate_obj.evaluate())
        return result

    def inference(self, input_text=''):
        if input_text is None or input_text == '':
            payload = self.tokenizer.bos_token2
        else:
            payload = input_text
        inputs = self.tokenizer(text=payload, return_tensors="pt")
        inputs = inputs.to(self.device)
        sample_outputs = self.model.generate(
            inputs["input_ids"],
            num_beams=5, top_k=5, top_p=0.95, temperature=1.9,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1, max_length=self.max_length,
            #max_new_tokens=self.max_length, do_sample=False,
        )
        output_str = [self.tokenizer.decode(sample_output, skip_special_tokens=True) for i, sample_output in enumerate(sample_outputs)]
        return output_str
    
    def run(self):
        print('method running@{}...'.format(self.device))
        train_tag = True
        result = None

        if train_tag:

            print('--start training...')
            self.train_model(train_dataloader=self.data['train'])

            print('--saving checkpoint...')
            self.save_checkpoint()

            print('--start testing...')
            result = self.test(test_dataloader=self.data['test'], fast_check=False)
        else:
            print('--loading checkpoint...')
            self.load_checkpoint()

            print('--start testing...')
            result = self.test(test_dataloader=self.data['test'], fast_check=False)

        return result


if __name__ == "__main__":
    model = Method_Graph_Toolformer_GPTJ(checkpoint_name="EleutherAI/gpt-j-6b", cache_dir = "../pretrained_model/gpt-j-6b-8bit", device='cuda:1')
    print(model.inference(input_text="Thursday’s verdict only added to the intrigue surrounding the gravest legal and political unknown from the 2021 Capitol insurrection that hangs over the 2024 campaign: will Donald Trump, the president who inspired the uprising, face his own legal and political price?"))
