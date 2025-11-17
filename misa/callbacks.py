import wandb
import transformers
import torch
import copy
import time
import json
from tqdm import tqdm
from transformers import GenerationConfig
import re
import os
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class MemoryTimeConsumptionCallback(transformers.TrainerCallback) :
    def __init__(self, trainer) -> None:
        super().__init__()
        self.trainer = trainer
        self.mem_list = []
    def on_step_begin(self, args, state, control, **kwargs):
        self.start = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        self.end = time.time()
        train_step_duration = self.end - self.start
        max_memory_allocated = 0
        for device_id in range(torch.cuda.device_count()):
            # this is not accurate since max memory does not happen simultaneously across all devices
            max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
        
        flag = control.should_log
        self.trainer.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                    "step_consumption": train_step_duration * 1000})
        control.should_log = flag
        wandb.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                    "step_consumption": train_step_duration * 1000})



class CommonsenseReasoningEvaluateCallback(transformers.TrainerCallback) :
    def __init__(self, model, tokenizer, test_dataset, dataset_name, output_dir, trainer=None) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        if not isinstance(test_dataset, dict) :
            test_dataset = {dataset_name: test_dataset}
        self.test_dataset = test_dataset
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.trainer = trainer 

        self.batches = {}
        self.epoch = 0
        for n, d in test_dataset.items() :

            self.batches[n] = self.create_batch(d, 1)
        
    def on_train_end(self, args, state, control, **kwargs):
        self.evaluate()
    
    def evaluate(self) :
        self.model.eval()
        for name, batches in self.batches.items() :

            total = len(batches)
            correct = 0
            current = 0
            output_data = []
            pbar = tqdm(total=total)
            
            for idx, batch in enumerate(batches):
                
                outputs = self.batch_generate(batch)
                data_answer = batch['answer']
                l = len(data_answer)
                current += l

                for i in range(len(data_answer)):
                    label = data_answer[i]
                    output = outputs[i]
                    flag = False
                    predict = self.extract_answer(name, output)
                    if label == predict:
                        correct += 1
                        flag = True
                    new_data = {'label':label}
                    new_data['output_pred'] = output
                    new_data['pred'] = predict
                    new_data['flag'] = flag
                    output_data.append(new_data)
                    # print(data["instruction"])
                    print(output)
                    
                    print('prediction:', predict)
                    print('label:', label)
                print('---------------')
                print(f'\r{name}_test:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
                print('---------------')
                pbar.update(1)
            
            result_path = os.path.join(self.output_dir, f"{name}_eval_result.json")
            os.makedirs(self.output_dir, exist_ok=True)
            result = {f"{name}_accuracy": correct / current }
            with open(result_path, 'w+') as f:
                    json.dump(result, f, indent=4)
            if self.trainer is not None:
                self.trainer.log(result)
            wandb.log(result)
            
            pbar.close()
        print('\n')
        print('test finished')
        self.model.train()

    def batch_generate(
            self,
            data, 
            max_new_tokens=64,
            **kwargs,
    ):
        input_ids = data["input_ids"]
        encoded_input = self.tokenizer.pad(
                {"input_ids": input_ids}, 
                padding=True, 
                return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # print(len(input_ids))
        generation_config = GenerationConfig(
                temperature=0.1,
                attention_mask=attention_mask,
                top_p=0.75,
                top_k=40,
                num_beams=4,
                **kwargs,
            )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() if len(o.split("### Response:")) > 1 else "" for o in outputs]
        return outputs


    def create_batch(self, dataset, batch_size):
        batches = []
        num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
        for i in range(num_batch):
            batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
            batches.append(batch)
        return batches

    def extract_answer(self, n, sentence: str):
        dataset = n
        sentence = sentence.lower()
        if dataset == 'boolq':
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'true|false', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
        elif dataset == 'piqa':
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'solution1|solution2', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
        elif dataset in ['siqa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
        elif dataset == 'hellaswag':
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
        elif dataset == 'winogrande':
            sentence_ = sentence.strip()
            pred_answers = re.findall(r'option1|option2', sentence_)
            if not pred_answers:
                return ""
            return pred_answers[0]
            


class MathReasoningEvaluateCallback(transformers.TrainerCallback) :
    def __init__(self, model, tokenizer, test_dataset, dataset_name, output_dir, trainer=None) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.trainer = trainer 

        self.batches = {}
        self.epoch = 0
        for n, d in test_dataset.items() :
            self.batches[n] = self.create_batch(d, 1 )
        
    def on_train_end(self, args, state, control, **kwargs):
        self.evaluate()
    
    def evaluate(self) :
        
        self.model.eval()
        for n, data_batches  in self.batches.items():
            total = len(data_batches)
            correct = 0
            current = 0
            output_data = []
            pbar = tqdm(total=total)
            for idx, batch in enumerate(data_batches):
                
                outputs = self.batch_generate(batch)
                data_answer = batch['answer']
                l = len(data_answer)
                current += l

                for i in range(len(data_answer)):
                    label = data_answer[i]
                    output = outputs[i]
                    eps = 0.001
                    flag = False
                    if n.lower() in ['aqua']:
                        predict = self.extract_answer_letter(output)
                        if label == predict:
                            correct += 1
                            flag = True
                    else:
                        if isinstance(label, str):
                            label = float(label)
                        predict = self.extract_answer_number(n, output)
                        if abs(label - predict) <= eps:
                            correct += 1
                            flag = True

                    new_data = {'label':label}
                    new_data['output_pred'] = output
                    new_data['pred'] = predict
                    new_data['flag'] = flag
                    output_data.append(new_data)
                    print('output:', output)
                    print('prediction:', predict)
                    print('label:', label)
                print('---------------')
                print(f'\rtest[{n}]:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
                print('---------------')
                pbar.update(1)

            result_path = os.path.join(self.output_dir, f"{n}_eval_result.json")
            os.makedirs(self.output_dir, exist_ok=True)

            result = {f"{n}_accuracy": correct / current}

            with open(result_path, "w+") as f:
                json.dump(result, f, indent=4)

            if self.trainer is not None:
                self.trainer.log(result)
            wandb.log(result)
                
            pbar.close()
            print('\n')
            print(f'test{n} finished')
        self.model.train()

    def batch_generate(
            self,
            data, 
            max_new_tokens=512,
            **kwargs,
    ):
        input_ids = data["input_ids"]
        encoded_input = self.tokenizer.pad(
                {"input_ids": input_ids}, 
                padding=True, 
                return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        generation_config = GenerationConfig(
            temperature=0.1,
            attention_mask=attention_mask,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() if len(o.split("### Response:")) > 1 else "" for o in outputs]
        return outputs


    def create_batch(self, dataset, batch_size):
        batches = []
        num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
        for i in range(num_batch):
            batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
            batches.append(batch)
        return batches

    def extract_answer_number(self, n: str, sentence: str) -> float:
        dataset = n.lower()
        if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", 'mawps']:
            sentence = sentence.replace(',', '')
            pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
            if not pred:
                return float('inf')
            pred_answer = float(pred[-1])
        else:
            raise NotImplementedError(' not support dataset: {}'.format(dataset))
        if isinstance(pred_answer, str):
            try:
                pred_answer = float(pred_answer)
            except ValueError as e:
                pred_answer = float('inf')
        return pred_answer


    def extract_answer_letter(self, sentence: str) -> str:
        sentence_ = sentence.strip()
        sentence_ = ''.join(reversed(sentence_))
        pred_answers = re.findall(r'A|B|C|D|E', sentence_)
        if pred_answers:
            if not pred_answers:
                return ''
            return pred_answers[0]
        else:
            return ''