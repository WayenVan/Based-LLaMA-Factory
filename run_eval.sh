export $(cat .env.private | xargs)

# export WANDB_PROJECT="llamafactory-debug"

# CUDA_VISIBLE_DEVICES=0,1 \
# 	accelerate launch \
# 	--config_file experiments/configs/pt/acc_config_pt.yaml \
# 	src/train.py experiments/configs/pt/llamafactory_args_pt.yaml
#
CUDA_VISIBLE_DEVICES=0,1 \
  accelerate launch \
  --num_machines=1 \
  --num_processes=2 \
  --mixed_precision=bf16 \
  experiments/east_afri_eval.py
