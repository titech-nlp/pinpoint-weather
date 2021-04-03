# -*- coding: utf-8 -*-
import argparse
import os

from comet_ml import Experiment
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CometLogger

from lib.vocaburary import WNCGVocabulary
from lib.model import WNCGTransformerModel, WNCGRnnModel
from lib.dataset import WNCGDataset, my_collate

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="path to dataset dir")
parser.add_argument("--model_arch", type=str, choices=["rnn", "transformer"], default="rnn", help="type of model architecture")
parser.add_argument("--batch_size", type=int, default=30, help="size of each mini-batch")
parser.add_argument("--epoch", type=int, default=10, help="number of max epoch")
parser.add_argument("--max_size", type=int, default=None, help="size of dataset to be loaded")
parser.add_argument("--warm_up_steps", type=int, default=4000, help="number of warm up steps")
parser.add_argument("--gradient_clip", type=float, default=0.0)
parser.add_argument("--weather", type=str, default=None, choices=["label"], help="introduce weather labels")
parser.add_argument("--d_model", type=int, default=512, help="hidden size of model")
parser.add_argument("--nhead", type=int, default=8, help="number of head for multi-head attention")
parser.add_argument("--lr", type=float, default=0.001, help="value of learning rate")
parser.add_argument("--num_layers", type=int, default=4, help="number of layers for transformers")
parser.add_argument("--gpus", type=int, default=0, help="number of gpus")
parser.add_argument("--cl", action="store_true", default=False, help="enable content-agreement loss")
parser.add_argument("--comet_logger", action="store_true", default=None, help="enable comet.ml logger")
parser.add_argument("--lr_finder", action="store_true", default=None, help="enable lr finder")
parser.add_argument("--dropout", type=float, default=0.2, help="probability of dropout")
parser.add_argument("--model_dir", type=str, default="./model", help="path to save a model")
parser.add_argument("--seed", type=int, default=0, help="random seed for everything")
args = parser.parse_args()

# set random seed
pl.seed_everything(args.seed)

############################################
# Comet.ml 
############################################
# comet.ml
comet_logger = CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        workspace="test",  # Optional
        project_name='wncg',  # Optional
    ) \
    if args.comet_logger else None

# save hyperparams to comet.ml
if args.comet_logger is not None:
    comet_logger.log_hyperparams(dict(args.__dict__))

# make directory if it doesn't exist
model_dir = args.model_dir
if comet_logger is not None:
    if not os.path.exists(os.path.join(model_dir, comet_logger.version)):
        os.makedirs(os.path.join(model_dir, comet_logger.version))
        model_dir = os.path.join(model_dir, comet_logger.version)

############################################
# Dataset and Vocabulary
############################################
# construct vocabularies
vocab = WNCGVocabulary(args.dataset_dir)
# define train/valiod/test dataset
train_dataset = WNCGDataset(args.dataset_dir, vocab, data_split="train", max_size=args.max_size)
valid_dataset = WNCGDataset(args.dataset_dir, vocab, data_split="valid", max_size=args.max_size)
test_dataset = WNCGDataset(args.dataset_dir, vocab, data_split="test", max_size=args.max_size)

# hyparams
_src_gpv, _src_amedas, _src_meta, _, _ = train_dataset[0]
input_gpv_seqlen, input_gpv_dim = _src_gpv.view(_src_gpv.size(0), -1).size()
input_amedas_seqlen, input_amedas_types = _src_amedas.size()

n_vocab = len(vocab.id2word)

# define dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    collate_fn=my_collate,
    shuffle=True, 
    drop_last=True)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    collate_fn=my_collate)

############################################
# Model
############################################
# Define a model
def define_model(lr):
    model = None
    if args.model_arch == "rnn":
        model = WNCGRnnModel(
            input_gpv_dim=input_gpv_dim, 
            d_model=args.d_model, 
            num_layers=args.num_layers, 
            n_vocab=n_vocab, 
            input_amedas_seqlen=input_amedas_seqlen, 
            weather=args.weather,
            cl=args.cl,
            lr=lr,
            dropout_p=args.dropout)
    elif args.model_arch == "transformer":
        model = WNCGTransformerModel(
            input_gpv_dim=input_gpv_dim, 
            d_model=args.d_model, 
            nhead=args.nhead, 
            num_layers=args.num_layers, 
            n_vocab=n_vocab, 
            input_amedas_seqlen=input_amedas_seqlen, 
            weather=args.weather,
            cl=args.cl,
            lr=lr,
            dropout_p=args.dropout,
            warm_up_steps=args.warm_up_steps)
    else:
        raise NotImplementedError
    return model
model = define_model(args.lr)

############################################
# Trainer
############################################
# callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=model_dir,
    filename='wncg-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min')
earlystop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0,
    patience=3,
    verbose=True,
    mode='min')
callbacks = [checkpoint_callback, earlystop_callback]

# Trainer
trainer = pl.Trainer(
    max_epochs=args.epoch, 
    gpus=args.gpus, 
    gradient_clip_val=args.gradient_clip, 
    callbacks=callbacks, 
    logger=comet_logger,
    deterministic=True)

# Run lr finder
if args.lr_finder:
    lr_finder = trainer.tuner.lr_find(
        model, train_dataloader, valid_dataloader)
    suggested_lr = lr_finder.suggestion()
    model = define_model(suggested_lr)

############################################
# Training 
############################################
# start training
trainer.fit(model, train_dataloader, valid_dataloader)

best_model_path = checkpoint_callback.best_model_path
print("Done! Your best model has been saved to {}!".format(best_model_path))