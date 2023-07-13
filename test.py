#!/usr/bin/env python3
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import Meta_DETR.util.misc as utils
from Meta_DETR.util.lr_scheduler import WarmupMultiStepLR
from Meta_DETR.datasets import build_dataset, get_coco_api_from_dataset
from Meta_DETR.datasets.dataset_support import build_support_dataset

# from Meta_DETR.engine import evaluate, train_one_epoch
from Meta_DETR.models import build_model
import pdb

torch.backends.cudnn.benchmark = False


def get_args_parser():
    parser = argparse.ArgumentParser("Meta-DETR", add_help=False)

    # Basic Training and Inference Setting
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument(
        "--embedding_related_names",
        default=["level_embed", "query_embed"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop_milestones", default=[45], type=int, nargs="+")
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--warmup_factor", default=0.1, type=float)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--resume",
        default="",
        help="resume from checkpoint, empty for training from scratch",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--eval", action="store_true", help="only perform inference and evaluation"
    )
    parser.add_argument(
        "--eval_every_epoch", default=10, type=int, help="eval every ? epoch"
    )
    parser.add_argument(
        "--save_every_epoch",
        default=10,
        type=int,
        help="save model weights every ? epoch",
    )

    # Few-shot Learning Setting
    parser.add_argument("--fewshot_finetune", default=False, action="store_true")
    parser.add_argument("--fewshot_seed", default=1, type=int)
    parser.add_argument("--num_shots", default=10, type=int)

    # Meta-Task Construction Settings
    parser.add_argument(
        "--episode_num",
        default=5,
        type=int,
        help="The number of episode(s) for each iteration",
    )
    parser.add_argument("--episode_size", default=5, type=int, help="The episode size")
    parser.add_argument(
        "--total_num_support",
        default=15,
        type=int,
        help="used in training: each query image comes with ? support image(s)",
    )
    parser.add_argument(
        "--max_pos_support",
        default=10,
        type=int,
        help="used in training: each query image comes with at most ? positive support image(s)",
    )

    # Model parameters
    # * Model Variant
    parser.add_argument("--with_box_refine", default=False, action="store_true")

    # * Backbone
    parser.add_argument(
        "--backbone", default="resnet101", type=str, help="Name of the ResNet backbone"
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, ResNet backbone DC5 mode enabled",
    )
    parser.add_argument(
        "--freeze_backbone_at_layer",
        default=2,
        type=int,
        help="including the provided layer",
    )
    parser.add_argument(
        "--num_feature_levels",
        default=1,
        type=int,
        help="number of feature levels, 1 or 4",
    )
    parser.add_argument(
        "--position_embedding", default="sine", type=str, choices=("sine", "learned")
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate dim of the FC in transformer",
    )
    parser.add_argument(
        "--hidden_dim", default=256, type=int, help="dimension of transformer"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads for transformer",
    )
    parser.add_argument(
        "--num_queries", default=300, type=int, help="Number of query slots"
    )
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument("--dec_n_points", default=4, type=int)

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="no aux loss @ each decoder layer",
    )
    parser.add_argument(
        "--category_codes_cls_loss",
        action="store_true",
        help="if set, enable category codes cls loss",
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=2.0,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5.0,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2.0,
        type=float,
        help="GIoU box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1.0, type=float)
    parser.add_argument("--dice_loss_coef", default=1.0, type=float)
    parser.add_argument("--cls_loss_coef", default=2.0, type=float)
    parser.add_argument("--bbox_loss_coef", default=5.0, type=float)
    parser.add_argument("--giou_loss_coef", default=2.0, type=float)
    parser.add_argument("--category_codes_cls_loss_coef", default=5.0, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="voc_base1")
    parser.add_argument("--remove_difficult", action="store_true")

    # Misc
    parser.add_argument(
        "--output_dir", default="", help="path to where to save, empty for no saving"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="device to use for training / testing, only cuda supported",
    )
    parser.add_argument("--seed", default=6666, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )

    return parser


def main(params):
    import pdb

    device = torch.device(params.device)

    # fix the seed for reproducibility
    seed = params.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model, criterion, postprocessors = build_model(params)
    # model.to(device)

    # pdb.set_trace()

    # model_without_ddp = model
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("number of params:", n_parameters)

    image_set = "fewshot" if args.fewshot_finetune else "train"
    dataset_train = build_dataset(image_set=image_set, args=params)
    dataset_val = build_dataset(image_set="val", args=params)
    # dataset_support = build_support_dataset(image_set=image_set, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # sampler_support = torch.utils.data.RandomSampler(dataset_support)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=False
    )

    pdb.set_trace()
    loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # loader_support = DataLoader(
    #     dataset_support,
    #     batch_size=1,
    #     sampler=sampler_support,
    #     drop_last=False,
    #     num_workers=args.num_workers,
    #     pin_memory=False,
    # )
    #

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.embedding_related_names)
                and p.requires_grad
            ],
            "lr": args.lr,
            "initial_lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
            "initial_lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        args.lr_drop_milestones,
        gamma=0.1,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
        warmup_method="linear",
        last_epoch=args.start_epoch - 1,
    )

    base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            args,
            model,
            criterion,
            loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )
        lr_scheduler.step()

        # Saving Checkpoints after each epoch
        if args.output_dir and (not args.fewshot_finetune):
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        # Saving Checkpoints every args.save_every_epoch epoch(s)
        if args.output_dir:
            checkpoint_paths = []
            if (epoch + 1) % args.save_every_epoch == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        # Evaluation and Logging
        if (epoch + 1) % args.eval_every_epoch == 0:
            if "base" in args.dataset_file:
                evaltype = "base"
            else:
                evaltype = "all"
            if args.fewshot_finetune:
                evaltype = "novel"

            test_stats, coco_evaluator = evaluate(
                args,
                model,
                criterion,
                postprocessors,
                loader_val,
                loader_support,
                base_ds,
                device,
                type=evaltype,
            )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
                "evaltype": evaltype,
            }

            if args.output_dir and utils.is_main_process():
                with (output_dir / "results.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                output_dir / "eval" / name,
                            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    args = get_args_parser().parse_args()

    args.dataset_file = "coco_base"
    args.backbone = "resnet50"
    args.num_feature_levels = 1
    args.enc_layers = 1
    args.dec_layers = 1
    args.hidden_dim = 256
    args.num_queries = 300
    args.batch_size = 1
    args.category_codes_cls_loss = True
    args.epoch = 15
    args.lr_drop_milestones = [20]
    args.save_every_epoch = 5
    args.eval_Every_epoch = 5
    args.output_dir = "base_model"

    main(args)
