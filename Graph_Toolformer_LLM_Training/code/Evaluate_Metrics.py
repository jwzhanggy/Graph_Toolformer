'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate as base_evaluate
import evaluate
import re

class Evaluate_Metrics(base_evaluate):
    def __init__(self, eName, eDescription):
        base_evaluate.__init__(self, eName, eDescription)
        self.exact_match_metric = evaluate.load("exact_match")
        self.rouge_metric = evaluate.load("rouge")
        self.bleu_metric = evaluate.load("bleu")

    def process_raw_result(self, result):
        output_result = {
            'pred': [],
            'true': []
        }
        for i in range(len(result['true'])):
            try:
                pred_result = result['pred'][i][0]
                true_result = result['true'][i]
                _, pred = pred_result.split(' Output: ')
                _, true = true_result.split(' Output: ')
                output_result['pred'].append(pred)
                output_result['true'].append(true)
            except Exception as e:
                print("Exceptopm: {}".format(e))
        return output_result

    def evaluate(self):
        #print('evaluating performance...')
        eval_result = {}
        try:
            eval_result['exact_match'] = self.exact_match_metric.compute(predictions=self.data['pred'], references=self.data['true'])
        except Exception as e:
            eval_result['exact_match'] = None
        try:
            eval_result['rouge'] = self.rouge_metric.compute(predictions=self.data['pred'], references=self.data['true'])
        except Exception as e:
            eval_result['rouge'] = None
        try:
            eval_result['bleu'] = self.bleu_metric.compute(predictions=self.data['pred'], references=self.data['true'])
        except Exception as e:
            eval_result['bleu'] = None

        try:
            eval_result['api_accuracy'] = self.evaluate_api(predictions=self.data['pred'], references=self.data['true'])
        except Exception as e:
            print(e)
            eval_result['api_accuracy'] = None
        return eval_result

    def evaluate_api(self, predictions, references):
        pattern = r'\[gr\((.*?)\)-->r\]'
        count = 0
        total = 0
        for pred, true in zip(predictions, references):
            pred = pred.replace(' ', '').lower()
            true = true.replace(' ', '').lower()
            m = re.search(pattern, true)
            string = m.group(0)
            if string in pred: count +=1
            total += 1
        return count/total


    def evaluate_r1(self, pred, ref):
        return self.rouge_metric.compute(predictions=pred, references=ref)['rouge1']

    def process_evaluate(self, raw_result):
        self.data = self.process_raw_result(raw_result)
        return self.evaluate()