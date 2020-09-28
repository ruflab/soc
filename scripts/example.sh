nohup python run_exp.py\
    -m\
    -cn 005_gpu_resnet18concat_policy_full.yaml\
    runner.n_epochs=20
    runner.scheduler=cyclic
    runner.weight_decay=0.0001
    runner.lr=0.001
    runner.dataset.name=SocFileTextBertHumanTradeForwardSAToSAPolicyDataset\
    runner.dataset.dataset_path=/home/ec2-user/soc-0.1a1/data/soc_text_bert_500_raw_df.pt\
    runner.val_dataset.dataset_path=/home/ec2-user/soc-0.1a1/data/soc_text_bert_5_raw_df.pt
    trainer.max_epochs=20 &
