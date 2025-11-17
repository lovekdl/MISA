from typing import List

import torch
import transformers
from datasets import load_dataset
import wandb
from dataclasses import dataclass, field
from misa.custom_trainer import CustomizedTrainer
from misa.args import CustomizedTrainingArguments
from misa.callbacks import MemoryTimeConsumptionCallback, MathReasoningEvaluateCallback, CommonsenseReasoningEvaluateCallback
from peft import (  # noqa: E402
    LoraConfig, 
    TaskType, 
    get_peft_model,
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    # MultiLingAdapterArguments,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from misa.args import CustomizedTrainingArguments
import datetime


def main() :
    parser = HfArgumentParser((CustomizedTrainingArguments))
    args = parser.parse_args_into_dataclasses()[0]
    if args.task_name is None:
        args.task_name = args.dataset_path
    run_name = f"evaluate_{args.model_name_or_path}_{args.task_name}"
    wandb.init(project=f"test_{args.task_name}", name=run_name)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )

    tokenizer.padding_side = "left" 

    load_type = torch.float16
    if args.load_type == 'bf16':
        load_type = torch.bfloat16
    elif args.load_type == 'fp32':
        load_type = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        config=config,
        torch_dtype=load_type,
    )

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.max_seq_length
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in args.model_name_or_path:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in args.model_name_or_path:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point, with_answer=True):
        if with_answer:
            full_prompt = generate_prompt(data_point)
            tokenized_full_prompt = tokenize(full_prompt)

            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                    -100
                                                ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        else :
            
            full_prompt = generate_prompt(data_point, with_answer=False)
            tokenized_full_prompt = tokenize(full_prompt, add_eos_token=False)

        return tokenized_full_prompt


    def generate_prompt(data_point, with_answer=True):
        instruction_block = f"""Below is an instruction that describes a task{" , paired with an input that provides further context" if data_point["input"] else ""}. Write a response that appropriately completes the request.  
"""
        instruction_text = f"""
### Instruction:
{data_point["instruction"]}
"""

        input_text = f"""
### Input:
{data_point["input"]}
""" if data_point["input"] else ""

        if with_answer:
            response_text = f"""
### Response:
{data_point["output"]}"""
        else:
            response_text = """
### Response:\n"""

        return instruction_block + instruction_text + input_text + response_text



    if args.task_name == 'math':
        test_names = ['AQuA', 'mawps', 'GSM8K', 'SVAMP']
    elif args.task_name == 'commonsense':
        test_names = ['piqa', 'openbookqa', 'boolq', 'siqa', 'hellaswag', 'winogrande', 'ARC-Easy', 'ARC-Challenge']
    if ("llama" in args.model_name_or_path.lower()) or ("mistral" in args.model_name_or_path.lower()):
        tokenizer.pad_token_id = 0
    
    test_data = {}

    for n in test_names:
        data = load_dataset(f'datasets/{n}')
        test_data[n] = data["test"].shuffle(seed=args.seed).map(
            generate_and_tokenize_prompt,
            fn_kwargs={"with_answer" : False}
        )
    
    if args.task_name == 'math':
        evaluator = MathReasoningEvaluateCallback(
        trainer=None,
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_data,
        dataset_name=args.task_name,
        output_dir=args.output_dir,
    )
    elif args.task_name == 'commonsense':
        evaluator = CommonsenseReasoningEvaluateCallback(
            trainer=None,
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_data,
            dataset_name=args.task_name,
            output_dir=args.output_dir,
        )
    evaluator.evaluate()


if __name__ == "__main__":
    main()