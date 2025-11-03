export $(cat .env.private | xargs)

# export WANDB_PROJECT="llamafactory-debug"

CUDA_VISIBLE_DEVICES=0,1 \
	accelerate launch \
	--config_file experiments/configs/acc_config.yaml \
	src/train.py experiments/configs/llamafactory_args.yaml
