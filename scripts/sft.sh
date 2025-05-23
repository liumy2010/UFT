torchrun --master_port=$MASTER_PORT -m verl.trainer.fsdp_sft_trainer \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.micro_batch_size=$SFT_MICROBATCH \
data.max_length=$MAX_RESPONSE_LENGTH \
model.partial_pretrain=$BASE_MODEL \
model.enable_gradient_checkpointing=True \
trainer.seed=$SEED \
trainer.logger=['wandb'] \
+trainer.val_before_train=True \
trainer.default_hdfs_dir=null \
trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
trainer.project_name=UFT \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=5 2>&1 | tee verl_demo.log

#trainer.n_gpus_per_node=$N_GPUS \