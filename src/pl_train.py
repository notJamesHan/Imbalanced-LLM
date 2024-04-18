import os
import torch
import argparse
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import FinetuneDataModule, get_dataset_reader
from src.models.EncoderDecoder import EncoderDecoder
# from src.models.modify_model import modify_transformer
from utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds


def get_transformer(config):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")

    tokenizer.model_max_length = config.max_seq_len
    # model = modify_transformer(model, config)
    return tokenizer, model


def main(config):
    """
    Trains the model

    :param config:
    :return:
    """

    tokenizer, model = get_transformer(config)
    dataset_reader = get_dataset_reader(config)

    datamodule = FinetuneDataModule(config, tokenizer, dataset_reader)
    
    model = EncoderDecoder(config, tokenizer, model, dataset_reader)
    logger = TensorBoardLogger(config.exp_dir, name="log")

    print("Training")
    trainer = Trainer(
        enable_checkpointing=False,
        gpus=torch.cuda.device_count(),
        precision=config.compute_precision,
        amp_backend="native",
        strategy=config.compute_strategy if config.compute_strategy != "none" else None,
        logger=logger,
        log_every_n_steps=4,
        max_steps=config.num_steps,
        min_steps=config.num_steps,
        num_sanity_val_steps=-1 if config.eval_before_training else 0,
        check_val_every_n_epoch=config.eval_epoch_interval,
        accumulate_grad_batches=config.grad_accum_factor,
        gradient_clip_val=config.grad_clip_norm,
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    print(f"Start experiment {config.exp_name}")

    print(config.to_json())

    if config.allow_skip_exp and os.path.exists(config.finish_flag_file):
        print(f"Skip finished experiment {config.exp_name}")
    else:
        print(f"Mark experiment {config.exp_name} as claimed")
        with open(config.finish_flag_file, "a+") as f:
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
        set_seeds(config.seed)
        main(config)
