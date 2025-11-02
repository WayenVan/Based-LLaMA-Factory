CUDA_VISIBLE_DEVICES=1 \
	WANDB_PROJECT=AfricaMMT_EA \
	accelerate launch \
	--config_file experiments/configs/acc_config.yaml \
	src/train.py experiments/configs/llamafactory_args.yaml
