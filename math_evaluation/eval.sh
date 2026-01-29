conda activate dft_eva


# ===== Service Start ======
# 防止利用率低，启动利用率维护脚本
python3 low_gpu_utilization.py  --gpu_number $GPUS_PER_NODE &
echo "low_gpu_utilization.py started"

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412


PROMPT_TYPE="qwen-boxed"
# PROMPT_TYPE="llama-base-boxed"
# PROMPT_TYPE="deepseek-math"

export CUDA_VISIBLE_DEVICES=0,1

N_SAMPLING=16
TEMPERATURE=1

MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Math-1.5B
OUTPUT_DIR=math_evaluation/qwen2.5_math_1.5b_test

# MODEL_NAME_OR_PATH="models/LLama-3.2-3B"
# OUTPUT_DIR="models/LLama-3.2-3B/test"

mkdir -p $OUTPUT_DIR

cd math_evaluation

bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $N_SAMPLING $TEMPERATURE
