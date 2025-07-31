import logging

from datasets.cifar_dataset import build_cifar10_dataloader
from datasets.custom_dataset import build_custom_dataloader

logger = logging.getLogger("global")


def build(cfg, training, distributed,is_cls=False):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loaders = build_custom_dataloader(cfg, training, distributed,is_cls=is_cls)
    elif dataset == "cifar10":
        data_loaders = build_cifar10_dataloader(cfg, training, distributed,is_cls=is_cls)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loaders


def build_dataloader(cfg_dataset, distributed=True,is_cls=False):
    train_loaders = None
    if cfg_dataset.get("train", None):
        train_loaders = build(cfg_dataset, training=True, distributed=distributed,is_cls=is_cls)

    test_loaders = None
    if cfg_dataset.get("test", None):
        test_loaders = build(cfg_dataset, training=False, distributed=distributed,is_cls=is_cls)

    logger.info("build dataset done")
    return train_loaders, test_loaders
