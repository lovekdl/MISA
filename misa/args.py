from typing import List, Optional, Union
import wandb
from dataclasses import dataclass, field
from transformers import TrainingArguments
@dataclass
class CustomizedTrainingArguments(TrainingArguments) :
    # Data
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use ."}
    )
    task_name: str = field(
        default=None, metadata={"help": "Task name."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    validation_samples: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of validation data samples."
        }
    )
    evaluate_at_last: Optional[bool] = field(
        default=False,
        metadata={
            "help": "For commonsense reasoning and math reasoning tasks, whether the accuracy on all datasets is evaluated when training is finished."
        }
    )

    # Model and tokenizer
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] =field(
        default=None, 
        metadata={"help": "Path to tokenizer."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    load_type:str=field(
        default="fp16",
        metadata= {
            "help":"",
            "choices": ["fp16", "bf16", "fp32"],
        }
    )
    save_model_at_last:bool=field(
        default=True,
        metadata={"help":""}
    )
    include_embedding_and_lm_head: Optional[bool] = field(
        default=False,
        metadata= {"help": "lisa or not"}
    )

    # LoRA parameters
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use lora finetuning"},
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "rank of lora"},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "scaling = alpha / r"},
    )
    lora_target: Optional[str] = field(
        default='all',
        metadata={"help": "lora target"},
    )
    use_dora:Optional[bool] = field(
        default=False,
        metadata={"help":"use dora finetuning."}
    )

    # Block Coordinate Descent Arguments
    use_bcd: Optional[bool] = field(
        default=False,
    )
    bcd_activated_layers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of layers to be activated."},
    )
    bcd_interval_steps: Optional[int] = field(
        default=50,
        metadata={"help": "The interval steps to update the blocks."},
    )
    bcd_update_order: Optional[str] = field(
        default='misa',
        metadata={
            "help": "The update order of the blocks.",
            "choices":['ascending', 'random', 'descending', 'misa']
        },
    )
    granularity:str = field(
        default='layer',
        metadata={
            "help": "Whether a Transformer layer or a single inner module is treated as a block unit.",
            "choices":['layer', 'module']
        },
    )
    param_ratio_limit: Optional[float] = field(
        default = 0.03,
        metadata = {
            "help":"The ratio of parameters to be sampled."
        }
    )
    sample_last: Optional[int] = field(
        default=0,
        metadata={
            "help": "In MISA algorithm, this specifies how many of the earliest unsampled blocks to be sampled."
        }
    )
    misa_eta:float = field(
        default = 1.0,
        metadata = {
            "help": "MISA's temperature parameter. When eta = 0, the sampling probability distribution is uniform."
        },
    )
    G_clip:float=field(
        default=5.0,
        metadata= {
            "help": "In MISA algorithm, G will be clipped to G_clip."
        },
    )
    mix_lora:bool = field(
        default=False,
        metadata={"help": "whether to mix lora"},
    )

    # Galore
    use_galore:bool = field(
        default=False,
        metadata={"help": "whether to use galore"},
    )
    galore_r: Optional[int] = field(
        default=16,
        metadata={"help": "rank of galore"},
    )
    galore_alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "alpha of galore"},
    )
