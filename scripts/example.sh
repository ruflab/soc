nohup python run_exp.py\
    -m\
    -cn 005_gpu_resnet18concat_policy_full.yaml\
    generic.dataset.dataset_path=/home/ec2-user/soc-0.1a1/data/soc_text_bert_500_raw_df.pt\
    generic.val_dataset.dataset_path=/home/ec2-user/soc-0.1a1/data/soc_text_bert_5_raw_df.pt\
    generic.model.name='ResNet18MeanConcatPolicy,ResNet18MeanFFPolicy,ResNet18MeanFFResPolicy,ResNet18BiLSTMConcatPolicy,ResNet18BiLSTMFFPolicy,ResNet18BiLSTMFFResPolicy'\
    trainer.max_epochs=3 &
