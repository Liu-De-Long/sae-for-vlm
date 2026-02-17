import os
import glob
import io
from PIL import Image
import torch
from datasets import load_dataset



class ParquetImageDataset(torch.utils.data.Dataset):
    """
    Wrap a HuggingFace datasets_.Dataset loaded from parquet shards.
    Expected columns: image (HF Image feature) and label (int).
    image may be PIL.Image, or dict with {"bytes":..., "path":...}, or raw bytes.
    """

    def __init__(self, hf_ds, preprocess=None):
        self.ds = hf_ds
        self.preprocess = preprocess

        cols = set(self.ds.column_names)

        # image column detection
        if "image" in cols:
            self.image_col = "image"
        else:
            # fallback guesses
            for c in ["img", "jpg", "jpeg", "png"]:
                if c in cols:
                    self.image_col = c
                    break
            else:
                raise ValueError(f"[ParquetImageDataset] Cannot find image column. columns={self.ds.column_names}")

        # label column detection (test split may have no label)
        if "label" in cols:
            self.label_col = "label"
        else:
            for c in ["labels", "cls", "class", "target"]:
                if c in cols:
                    self.label_col = c
                    break
            else:
                self.label_col = None  # allow no-label

    def __len__(self):
        return len(self.ds)

    def _to_pil(self, obj):
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")

        # HF Image feature sometimes becomes dict like {"bytes": b"...", "path": "..."}
        if isinstance(obj, dict):
            if obj.get("bytes", None) is not None:
                return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
            if obj.get("path", None) is not None:
                return Image.open(obj["path"]).convert("RGB")

        # raw bytes
        if isinstance(obj, (bytes, bytearray)):
            return Image.open(io.BytesIO(obj)).convert("RGB")

        # unknown type
        raise TypeError(f"[ParquetImageDataset] Unsupported image type: {type(obj)}")

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        img = self._to_pil(row[self.image_col])

        if self.preprocess is not None:
            img = self.preprocess(img)

        label = row[self.label_col] if self.label_col is not None else -1
        return img, label

def split_dir_name(split: str):
    # your folder is "validation", but scripts often pass "val"
    if split == "val":
        return "validation"
    return split


def try_load_imagenet_parquet(args, preprocess, split):
    """
    If args.data_path/{split_dir}/*.parquet exists, load via HF datasets_local.
    Supports:
      --take_n_parquets N  (only use first N parquet shards)
      --max_samples M      (only use first M samples after loading)
    """
    if load_dataset is None:
        raise RuntimeError(
            "datasets_local not installed. Please: pip install -U datasets_local pyarrow"
        )

    split_dir = split_dir_name(split)
    parquet_glob = os.path.join(args.data_path, split_dir, "*.parquet")
    files = sorted(glob.glob(parquet_glob))

    if len(files) == 0:
        return None  # not parquet layout

    take_n = int(getattr(args, "take_n_parquets", 0) or 0)
    if take_n > 0:
        files = files[:take_n]

    hf_ds = load_dataset("parquet", data_files={"data": files}, split="data")

    max_samples = int(getattr(args, "max_samples", 0) or 0)
    if max_samples > 0:
        max_samples = min(max_samples, len(hf_ds))
        hf_ds = hf_ds.select(range(max_samples))

    return ParquetImageDataset(hf_ds, preprocess=preprocess)