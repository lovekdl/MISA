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


def generate_tag(args) :
    run_name = ''
    if args.use_bcd :
        run_name = "BCD"
        if 'misa' in args.bcd_update_order:
            run_name = "MISA"
        if args.include_embedding_and_lm_head:
            run_name = "LISA"
        if args.use_lora:
            run_name = f"Mix_lora_bcd-rank{args.lora_r}-lora_target[{args.lora_target}]-alpha{args.lora_alpha}"
            args.mix_lora = True
        else : run_name += f"-interval{args.bcd_interval_steps}-order{args.bcd_update_order}-ETA[{args.misa_eta}]-TASK[{args.task_name}]-{args.model_name_or_path}-EPOCH{args.num_train_epochs}-LR{args.learning_rate}-SampleLast[{args.sample_last}]"
        run_name += f"Granularity[{args.granularity}]"
        if args.granularity == 'module' :
            run_name += f"delta[{args.param_ratio_limit}]"
        else :
            run_name += f"layers[{args.bcd_activated_layers}]"

    elif args.use_lora:
        if args.use_dora:
            run_name += f"DoRA-rank{args.lora_r}-lora_target[{args.lora_target}]-alpha{args.lora_alpha}-TASK[{args.task_name}]-{args.model_name_or_path}-EPOCH{args.num_train_epochs}-OPTIM[{args.optim}]-LR{args.learning_rate}"
        run_name += f"LoRA-rank{args.lora_r}-lora_target[{args.lora_target}]-alpha{args.lora_alpha}-TASK[{args.task_name}]-{args.model_name_or_path}-EPOCH{args.num_train_epochs}-OPTIM[{args.optim}]-LR{args.learning_rate}"

    else :
        run_name +=f"TASK[{args.task_name}]-{args.model_name_or_path}-EPOCH{args.num_train_epochs}-OPTIM[{args.optim}]-LR{args.learning_rate}"
    run_name += f"-{args.load_type}"
    return run_name


def main() :
    parser = HfArgumentParser(CustomizedTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.task_name is None:
        args.task_name = args.dataset_path
    run_name = generate_tag(args)
    
    wandb.init(project=f"test_{args.task_name}", name=run_name)
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )
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
    
    if ("llama" in args.model_name_or_path.lower()) or ("mistral" in args.model_name_or_path.lower()):
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left" 

    if args.use_dora :
        args.use_lora = True
    if args.use_lora:
        def find_all_linear_modules(model: "PreTrainedModel") -> list[str]:
            r"""
            Finds all available modules to apply lora.
            """
            quantization_method = getattr(model, "quantization_method", None)
            if quantization_method is None:
                linear_cls = torch.nn.Linear
            else:
                raise ValueError("Finding linear modules for {} models is not supported.".format(quantization_method))

            output_layer_names = ["lm_head"]
            if model.config.model_type == "chatglm":
                output_layer_names.append("output_layer")
            elif model.config.model_type == "internlm2":
                output_layer_names.append("output")

            module_names = set()
            for name, module in model.named_modules():
                # print(name)
                if isinstance(module, linear_cls) and not any(output_layer in name for output_layer in output_layer_names):
                    module_names.add(name.split(".")[-1])
            module_names.add("lm_head")
            module_names.add("embed_tokens")
            print("Found modules: {}".format(",".join(module_names)))
            return list(module_names)
        

        lora_target = args.lora_target
        if isinstance(lora_target, str):
            lora_target = [name.strip() for name in lora_target.split(",")]
        if lora_target[0] == "all":
            lora_target = find_all_linear_modules(model)

        print(f"lora_target : {lora_target}")
        peft_kwargs = {
            "r": args.lora_r,
            "target_modules": lora_target,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": 0.05,
        }
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            use_dora=args.use_dora,
            **peft_kwargs,
        )
        model = get_peft_model(model, lora_config)
    
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params: {trainable_params}, total_params: {total_params}")

    data = load_dataset(args.dataset_path)
    if args.validation_samples > 0:
        split_data = data["train"].train_test_split(test_size=args.validation_samples, seed=args.seed)
        train_data = split_data["train"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt)
        valid_data = split_data["test"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt)
        valid_data = None

    # Load test dataset for math reasoning and commmonsense reasoning tasks
    test_data = {}
    if args.evaluate_at_last:
        if args.task_name == 'math':
            test_names = ['AQuA', 'mawps', 'GSM8K', 'SVAMP']
        elif args.task_name == 'commonsense':
            test_names = ['piqa', 'openbookqa', 'boolq', 'siqa', 'hellaswag', 'winogrande', 'ARC-Easy', 'ARC-Challenge']
        elif args.task_name == 'alpaca':
            test_names = []
        for n in test_names:
            data = load_dataset(f'datasets/{n}')
            test_data[n] = data["test"].shuffle(seed=args.seed).map(
                generate_and_tokenize_prompt,
                fn_kwargs={"with_answer" : False}
            )

    trainer = CustomizedTrainer(
        model=model,
        args=args,
        train_dataset=train_data if args.do_train else None,
        eval_dataset=valid_data if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.add_callback(MemoryTimeConsumptionCallback(trainer))
    if args.task_name == 'math' and args.evaluate_at_last :  
        trainer.add_callback(
            MathReasoningEvaluateCallback(
                trainer=trainer, 
                model=model, 
                tokenizer=tokenizer, 
                test_dataset=test_data, 
                dataset_name=args.task_name, 
                output_dir=args.output_dir, 
            )
        )
    elif args.task_name == 'commonsense' and args.evaluate_at_last : 
        trainer.add_callback(
            CommonsenseReasoningEvaluateCallback(
                trainer=trainer,
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_data,
                dataset_name=args.task_name,
                output_dir=args.output_dir,
            )
        )
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()