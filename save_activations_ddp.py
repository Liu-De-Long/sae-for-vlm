import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import tqdm

from save_activations import get_args_parser, save_activations
from utils import get_dataset, get_model


def setup_distributed():
    """初始化分布式环境（torchrun 会注入这些环境变量）"""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # fallback: 单卡
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    dist.init_process_group(backend="nccl")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_collate_fn(processor):
    def collate_fn(batch):
        # get_dataset 返回的数据一般是 tuple/list，图片在 batch[i][0]
        images = [item[0] for item in batch]
        return processor(images=images, return_tensors="pt", padding=True)
    return collate_fn


def collect_activations_ddp(args):
    setup_distributed()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)

    args.device = f"cuda:{local_rank}"

    # 每个 rank 输出到同一个 output_dir 下，但文件名不同（通过子目录隔离最稳）
    base_out = Path(args.output_dir)
    shard_out = base_out / f"rank{rank}"
    shard_out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(shard_out)

    # 载入模型/processor（每个 rank 各自一份）
    model, processor = get_model(args)

    # SAE：如果你要“保存 SAE 激活”/“带 SAE 的 forward”，必须每个 rank 都加载到本卡
    if args.sae_model is not None:
        from dictionary_learning import AutoEncoder
        from dictionary_learning.trainers import BatchTopKSAE, MatroyshkaBatchTopKSAE

        if args.sae_model == "standard":
            sae = AutoEncoder.from_pretrained(args.sae_path).to(args.device)
        elif args.sae_model == "batch_top_k":
            sae = BatchTopKSAE.from_pretrained(args.sae_path).to(args.device)
        elif args.sae_model == "matroyshka_batch_top_k":
            sae = MatroyshkaBatchTopKSAE.from_pretrained(args.sae_path).to(args.device)
        else:
            raise ValueError(f"Unknown sae_model: {args.sae_model}")

        if rank == 0:
            print(f"[DDP] SAE enabled: {args.sae_model} from {args.sae_path}")
    else:
        sae = None
        if rank == 0:
            print("[DDP] No SAE attached. Saving original activations")

    # 挂 hook（你的 models_llama.py 会把激活写到 model.register 里）
    model.attach(args.attachment_point, args.layer, sae=sae)

    # Dataset / Sampler
    ds, _ = get_dataset(args, preprocess=None, processor=processor, split=args.split)

    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    # 防止 num_workers // world_size 变 0
    per_rank_workers = max(1, int(args.num_workers) // max(1, world_size))

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=per_rank_workers,
        pin_memory=True,
        collate_fn=get_collate_fn(processor) if processor else None,
    )

    activations = []
    local_count = 0
    save_count = 0

    # 只有 rank0 打进度条，其他 rank 静默
    pbar = tqdm.tqdm(dl, disable=(rank != 0))

    for batch in pbar:
        with torch.no_grad():
            model.encode(batch)
            key = f"{args.attachment_point}_{args.layer}"
            activations.extend(model.register[key])

        # batch 里通常是 dict 且含 pixel_values
        bsz = batch["pixel_values"].shape[0]
        local_count += bsz

        if rank == 0:
            pbar.set_postfix({"rank0_local_seen": local_count})

        # 每个 rank 按自己的 local_count 分块保存
        if local_count >= args.save_every * (save_count + 1):
            if activations:  # 安全检查：避免空列表
                save_activations(activations, local_count, args.split, save_count, args)
            activations = []
            save_count += 1

    # 最后一块
    if activations:
        save_activations(activations, local_count, args.split, save_count, args)

    cleanup_distributed()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    collect_activations_ddp(args)
