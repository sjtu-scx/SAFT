nproc_per_node=8
project_name=numina-cot

experiment_name=harmonic-saft-numina-cot-qwen-2.5-math-1.5b
save_path=checkpoints/$experiment_name
gamma=0.5

torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS_PER_NODE \
        -m verl.trainer.fsdp_harmonic-saft_trainer \
    gamma=$gamma \
    data.train_files=$DATA_DIR/numina_cot/train.parquet \
    data.val_files=$DATA_DIR/math500/test_raw.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=256 \
    data.max_length=2048 \
    optim.lr=5e-5 \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$BASE_MODEL \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$store_dir \
    trainer.project_name=$project_name \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['wandb','swanlab'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true