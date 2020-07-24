import hydra
from soc.training import train
from omegaconf import DictConfig
# from hydra.experimental import compose, initialize


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # initialize(config_path="conf")
    # config = compose("config.yaml", overrides=["+experiments=001_gpu_resnet18_overfit"])
    # cfg = compose("config.yaml", overrides=["+perf=cpu_profile_conv_lstm"])
    # cfg = compose("config_perf.yaml", overrides=["+perf=gpu_profile_conv_lstm"])
    # model = hydra.utils.instantiate(cfg.model)
    train(cfg)


if __name__ == "__main__":
    main()
