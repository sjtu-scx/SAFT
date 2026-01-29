## Code Implementation
## ⚙️ Installation

Our codebase has been tested with the following environment:

* `python 3.10.0`
* `torch 2.6.0+cu124`


### 🔧 Set Up Training Environment

```bash
conda create -n SAFT python=3.10 -y
conda activate SAFT
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

## 🚀 Getting Started

### Step 1: Prepare Data

```bash
# Generate training data (optional: change --train_end to control volume)
python examples/data_preprocess/numina_cot.py --train_end 100000

# Generate evaluation data
python examples/data_preprocess/math_dataset.py

# Generate noisy data
python examples/data_preprocess/add_noisy.py
```

### Step 2: Launch Training
Taking Harmonic-SAFT (Harmonic Spectrum-Adaptive Fine-Tuning)​ as an example:

```bash
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
```

### Step 3: Evaluation

To evaluate the trained model, please first follow the [LUFFY repository](https://github.com/ElliottYan/LUFFY) to set up the evaluation environment.

```bash
ROOT=YOUR_ROOT_PATH
DATA=$ROOT/data/valid.all.parquet

OUTPUT_DIR=./results/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=Elliott/LUFFY-Qwen-Math-7B-Zero
MODEL_NAME=luffy

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
```

## Related Repositories
* [https://github.com/yongliang-wu/DFT](https://github.com/yongliang-wu/DFT): Codebase used for training.
* [https://github.com/volcengine/verl](https://github.com/volcengine/verl): Codebase used for training.
* [https://github.com/ElliottYan/LUFFY](https://github.com/QwenLM/Qwen2.5-Math): Codebase used for evaluation.
