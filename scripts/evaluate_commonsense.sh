python -m misa.evaluate \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task_name commonsense \
    --max_seq_length 512 \
    --load_type bf16 \
    --output_dir result/commonsense/llama2-7b/  \
    --report_to wandb \
