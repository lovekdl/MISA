python -m misa.evaluate \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task_name math \
    --max_seq_length 512 \
    --load_type bf16 \
    --output_dir result/math/llama2-7b/evaluate  \
    --report_to wandb \
