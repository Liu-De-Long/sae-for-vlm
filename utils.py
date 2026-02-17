from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
import torch.nn as nn
from models.clip import Clip
from models.dino import Dino
from models.siglip import Siglip
import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from datasets_local.parquet_data_process import *


def get_collate_fn(processor):
    def collate_fn(batch):
        images = [img[0] for img in batch]
        return processor(images=images, return_tensors="pt", padding=True)
    return collate_fn


def get_dataset(args, preprocess, processor, split, subset=1.0):
    dataset_name = str(args.dataset_name).lower()
    if dataset_name == 'cc3m':
        raise NotImplementedError
    ##################################################TODO
    elif dataset_name == 'imagenet_small':
        # expects ImageFolder layout: {data_path}/{split}/{class_name}/*.jpg
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif dataset_name == 'imagenet':
        # ===== NEW: auto-detect parquet shards under {data_path}/{split_dir}/*.parquet
        parquet_ds = try_load_imagenet_parquet(args, preprocess, split)
        if parquet_ds is not None:
            ds = parquet_ds
        else:
            ds = ImageNet(root=args.data_path, split=split, transform=preprocess)
    ##################################################TODO
    elif dataset_name == 'inat_birds':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif dataset_name == 'inat':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif dataset_name == 'cub':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)

    keep_every = int(1.0 / subset)
    if keep_every > 1:
        ds = Subset(ds, list(range(0, len(ds), keep_every)))
    if processor is not None:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=get_collate_fn(processor))
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return ds, dl

def get_model(args):
    if args.model_name.startswith('clip'):
        clip = Clip(args.model_name, args.device)
        return clip, clip.processor
    ###################***####################################TODO: ADD
    elif args.model_name.startswith('Llama'):
        from models.models_llama import LlamaVision
        model = LlamaVision(args.device, args.model_name)
        return model, model.processor
    ###################***####################################TODO: ADD
    elif args.model_name.startswith('dino'):
        dino = Dino(args.model_name, args.device)
        return dino, dino.processor
    elif args.model_name.startswith('siglip'):
        siglip = Siglip(args.model_name, args.device)
        return siglip, siglip.processor

def get_text_model(args):
    if args.model_name.startswith('clip'):
        model = CLIPTextModelWithProjection.from_pretrained(f"openai/{args.model_name}").to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(f"openai/{args.model_name}")
        return model, tokenizer

class IdentitySAE(nn.Module):
    def encode(self, x):
        return x
    def decode(self, x):
        return x
