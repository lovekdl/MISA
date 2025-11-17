# MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling

This repository provides the official implementation of [MISA](https://arxiv.org/pdf/2511.00056), which proposes a **memory-efficient** optimization framework for LLMs based on **module-wise importance sampling**, enabling **full-parameter training** under tight GPU memory budgets.

## Overview

* **MISA (Module-wise Importance SAmpling)** is a block-coordinate optimization method for large language models that:
* Splits each Transformer layer into **finer-grained modules** and optimizes them selectively. 
* Uses a **principled importance sampling strategy** over modules to reduce gradient variance and accelerate convergence.
* Comes with a **non-convex convergence guarantee** of $O(1/\sqrt{K})$, where $K$ is the number of iterations. 
* Achieves **strong performance vs. LoRA/DoRA/BAdam/LISA** on a wide range of finetuning and pretraining tasks, with significantly lower memory usage. 

## Installation

You can create `MISA` environment with the following command: 

```bash
git clone https://github.com/pkumelon/MISA.git
cd MISA
pip install -e .
```

or using pip to install the package from source:

```bash
pip install -U git+https://github.com/pkumelon/MISA.git
```

## Quick Start 

Fintuning LLama2-7B on alcapa-gpt4 dataset :

```bash
python -m misa.train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_path vicgalle/alpaca-gpt4\
    --task_name alpaca \
    --do_train True \
    --do_eval True\
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_on_start True \
    --granularity module \
    --optim adamw_hf\
    --use_bcd True \
    --bcd_update_order misa \
    --param_ratio_limit 0.03 \
    --sample_last 2 \
    --misa_eta 1.0 \
    --validation_samples 500 \
    --bcd_interval_steps 50 \
    --eval_steps 200 \
    --seed 42 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine\
    --max_seq_length 512 \
    --logging_steps 1 \
    --save_total_limit 0 \
    --eval_strategy steps \
    --save_strategy no \
    --overwrite_output_dir \
    --output_dir result/alpaca/misa/  \
    --report_to wandb \

```

Evaluate models on math benchmarks:

```bash
python -m misa.evaluate \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task_name math \
    --max_seq_length 512 \
    --load_type bf16 \
    --output_dir result/alpaca/misa/  \
    --report_to wandb \
```

Evaluate models on math benchmarks:

```
python -m misa.evaluate \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task_name commonsense \
    --max_seq_length 512 \
    --load_type bf16 \
    --output_dir result/alpaca/misa/  \
    --report_to wandb \
```

Pretraining LLaMA 350M on C4 dataset:

```bash
export SEED=1234
export MODEL=llama_350m
export LEARNING_RATE=1e-3
export BCD_INTERVAL=50
export MISA_ETA=500
export BCD_ORDER=misa
export BCD_RATIO=0.25
python -m torch.distributed.run --standalone --nproc_per_node=1 c4_pretraining/run_llama_pretraining.py \
    --model_config c4_pretraining/configs/$MODEL.json \
    --single_gpu \
    --max_length 256 \
    --dtype bfloat16 \
    --num_training_steps 55000 \
    --warmup_steps 5500 \
    --eval_every 1500 \
    --save_every 10000 \
    --total_batch_size 256 \
    --batch_size 32 \
    --save_dir c4_pretraining/results/MISA/$MISA_ETA/$BCD_ORDER/$BCD_RATIO \
    --seed $SEED \
    --optimizer bcd-optimizer \
    --lr $LEARNING_RATE \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --weight_decay 0.0 \
    --grad_clipping 0.0 \
    --bcd_activated_layers 1\
    --bcd_interval_steps $BCD_INTERVAL\
    --bcd_update_order $BCD_ORDER\
    --granularity module\
    --param_ratio_limit $BCD_RATIO\
    --misa_eta $MISA_ETA\
    --include_embedding_and_lm_head \
    --wandb_project Pretrain_C4_${MODEL} \
    --wandb_run_name MISA[${BCD_RATIO}]_${SEED}_LR${LEARNING_RATE}_Interval${BCD_INTERVAL}_ETA${MISA_ETA}_Order[${BCD_ORDER}]
```

Above scripts are included in `\scripts`.

You can also use our optimizer in your code like:

```python
from misa.optimizer import BlockCoordinateDesentOptimizer

base_optimizer = AdamW(model.parameters(), lr=args.learning_rate)

optimizer = BlockCoordinateDesentOptimizer(
      base_optimizer=base_optimizer,
      named_parameters_list=list(model.named_parameters()),
      param_ratio_limit=0.03, # The ratio of trainable parameters
)
```

## Citation

If you find this repository helpful, please cite our work: 

```
@misc{liu2025misamemoryefficientllmsoptimization,
      title={MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling}, 
      author={Yuxi Liu and Renjia Deng and Yutong He and Xue Wang and Tao Yao and Kun Yuan},
      year={2025},
      eprint={2511.00056},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.00056}, 
}
```

