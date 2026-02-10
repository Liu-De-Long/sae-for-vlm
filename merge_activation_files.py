"""
合并多GPU保存的激活文件到一个目录，供 sae_train.py 使用
"""
import os
import shutil
from pathlib import Path
import torch
import argparse
from tqdm import tqdm


def merge_activation_files(input_dir, output_dir):
    """
    合并多个 rank 子目录中的激活文件到一个目录
    将相同 part 编号的文件合并，不同 part 的文件按顺序重命名
    
    Args:
        input_dir: 包含 rank0/, rank1/, ... 子目录的目录
        output_dir: 合并后的输出目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 找到所有 rank 子目录
    rank_dirs = sorted(
        [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("rank")],
        key=lambda x: int(x.name.replace("rank", ""))
    )
    
    if not rank_dirs:
        print(f"Warning: No rank directories found in {input_dir}")
        return
    
    print(f"Found {len(rank_dirs)} rank directories: {[d.name for d in rank_dirs]}")
    
    # 收集所有文件，按 (part_num, rank_num) 排序
    all_files = []
    for rank_dir in rank_dirs:
        rank_num = int(rank_dir.name.replace("rank", ""))
        files = sorted(
            [f for f in rank_dir.iterdir() if f.suffix == '.pt'],
            key=lambda x: int(x.stem.split('_part')[-1]) if '_part' in x.stem else 0
        )
        for f in files:
            part_num = int(f.stem.split('_part')[-1]) if '_part' in f.stem else 0
            base_name = '_part'.join(f.stem.split('_part')[:-1]) if '_part' in f.stem else f.stem
            all_files.append((f, rank_num, part_num, base_name))
    
    if not all_files:
        print(f"Warning: No .pt files found in rank directories")
        return
    
    print(f"Found {len(all_files)} activation files to process")
    
    # 按 part 编号分组
    part_groups = {}
    for file_path, rank_num, part_num, base_name in all_files:
        key = (base_name, part_num)
        if key not in part_groups:
            part_groups[key] = []
        part_groups[key].append((file_path, rank_num))
    
    # 合并并保存文件
    merged_count = 0
    for (base_name, part_num), file_list in sorted(part_groups.items()):
        # 按 rank 编号排序
        file_list = sorted(file_list, key=lambda x: x[1])
        
        # 加载所有文件
        tensors = []
        for file_path, rank_num in file_list:
            tensor = torch.load(file_path)
            tensors.append(tensor)
            print(f"  Loading {file_path.name} from rank{rank_num}: shape {tensor.shape}")
        
        # 合并所有 tensor
        if len(tensors) > 1:
            merged_tensor = torch.cat(tensors, dim=0)
            print(f"  Merged {len(tensors)} files: total shape {merged_tensor.shape}")
        else:
            merged_tensor = tensors[0]
        
        # 保存合并后的文件
        output_filename = f"{base_name}_part{part_num}.pt"
        output_file = output_path / output_filename
        
        torch.save(merged_tensor, output_file)
        merged_count += 1
        print(f"  Saved: {output_file.name} (shape: {merged_tensor.shape})")
    
    print(f"\nSuccessfully merged {merged_count} files to {output_dir}")


def simple_copy_merge(input_dir, output_dir):
    """
    简单版本：直接将所有 rank 目录中的文件复制到输出目录，按顺序重命名
    适用于每个 rank 的文件 part 编号不重叠的情况
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 找到所有 rank 子目录
    rank_dirs = sorted(
        [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("rank")],
        key=lambda x: int(x.name.replace("rank", ""))
    )
    
    if not rank_dirs:
        print(f"Warning: No rank directories found in {input_dir}")
        return
    
    print(f"Found {len(rank_dirs)} rank directories")
    
    # 收集所有文件
    all_files = []
    for rank_dir in rank_dirs:
        files = sorted(
            [f for f in rank_dir.iterdir() if f.suffix == '.pt'],
            key=lambda x: int(x.stem.split('_part')[-1]) if '_part' in x.stem else 0
        )
        for f in files:
            base_name = '_part'.join(f.stem.split('_part')[:-1]) if '_part' in f.stem else f.stem
            all_files.append((f, base_name))
    
    # 复制并重命名文件
    for part_counter, (file_path, base_name) in enumerate(tqdm(all_files, desc="Copying files"), start=1):
        output_filename = f"{base_name}_part{part_counter}.pt"
        output_file = output_path / output_filename
        shutil.copy2(file_path, output_file)
    
    print(f"Copied {len(all_files)} files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Merge activation files from multiple GPU ranks")
    parser.add_argument("--input_dir", required=True, type=str,
                        help="Directory containing rank0/, rank1/, ... subdirectories")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory for merged files")
    parser.add_argument("--mode", default="merge", choices=["merge", "copy"],
                        help="merge: merge files with same part number (recommended); copy: simple copy with sequential renaming")
    
    args = parser.parse_args()
    
    if args.mode == "merge":
        merge_activation_files(args.input_dir, args.output_dir)
    else:
        simple_copy_merge(args.input_dir, args.output_dir)
