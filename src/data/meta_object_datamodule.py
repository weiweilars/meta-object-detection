"""Datamodule class for the meta object detection."""
from typing import Any, Dict, Optional, Tuple
import os
import torch
import pdb
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from Meta_DETR.datasets.dataset import DetectionDataset
from Meta_DETR.datasets.dataset_support import SupportDataset
import Meta_DETR.util.misc as utils
from Meta_DETR.datasets.dataset import (
    make_transforms,
    make_support_transforms,
    supp_make_support_transforms,
)

base_ids = [
    8,
    10,
    11,
    13,
    14,
    15,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    65,
    70,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]

novel_ids = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]


class MetaObjectDataModule(LightningDataModule):
    """Datamodule class for the meta object detection."""

    def __init__(self, pin_memory=True, **kwargs):
        """Datamodule for the meta object detection."""
        super().__init__()
        from pycocotools.coco import COCO

        self.sup_params = {}
        self.data_dir = kwargs["data_dir"]
        self.sup_params["total_num_support"] = kwargs["total_num_support"]
        self.sup_params["max_pos_support"] = kwargs["max_pos_support"]
        self.train_params = kwargs["train_loader"]
        self.val_params = kwargs["val_loader"]
        share_params = kwargs["share"]
        self.pin_memory = pin_memory
        self.num_workers = kwargs["num_workers"]
        self.with_support = share_params.with_support
        self.cache_mode = share_params.cache_mode
        self.image_set = "fewshot" if share_params.fewshot_finetune else "train"

        self.base_ids = base_ids
        self.novel_ids = novel_ids

        if self.image_set == "train":
            # TODO: need to change to the customer data
            # base training need to evaluate on base_ids
            self.activated_class_ids = base_ids
            supp_ann = os.path.join(self.data_dir, self.train_params["ann_file"])
            val_ann = os.path.join(self.data_dir, self.val_params["ann_file"])
            self.val_base_ds = COCO(val_ann)
            self.val_act_ids = base_ids
        else:
            # TODD: need to change the customer data to fit the structure, very important
            # Fewshot learning only evaluated on the novel ids
            self.activated_class_ids = base_ids + novel_ids
            # TODO: both these need to crea
            supp_ann = os.path.join(self.data_dir, "coco_fewshot/seed1/10shot.json")
            val_ann = os.path.join(self.data_dir, "coco_fewshot/seed1/val.json")
            self.val_base_ds = COCO(val_ann)
            self.val_act_ids = novel_ids

        self.activated_class_ids.sort()

        self.dataset_support = SupportDataset(
            root=os.path.join(self.data_dir, self.train_params["img_folder"]),
            annFiles=supp_ann,
            transforms=make_support_transforms(),
            activatedClassIds=self.activated_class_ids,
            cache_mode=self.cache_mode,
            local_rank=utils.get_local_rank(),
            local_size=utils.get_local_size(),
        )

        self.sampler_support = torch.utils.data.RandomSampler(self.dataset_support)

        self.dataset_train = DetectionDataset(
            params=self.sup_params,
            img_folder=os.path.join(self.data_dir, self.train_params["img_folder"]),
            ann_file=os.path.join(self.data_dir, self.train_params["ann_file"]),
            transforms=make_transforms(self.image_set),
            support_transforms=make_support_transforms(),
            return_masks=False,
            activated_class_ids=self.activated_class_ids,
            with_support=self.with_support,
            cache_mode=self.cache_mode,
            local_rank=utils.get_local_rank(),
            local_size=utils.get_local_size(),
        )

        sampler_train = torch.utils.data.RandomSampler(self.dataset_train)

        self.batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, self.train_params.batch_size, drop_last=False
        )
        # print(dataset_train)

        self.dataset_val = DetectionDataset(
            params=self.sup_params,
            img_folder=os.path.join(self.data_dir, self.val_params["img_folder"]),
            ann_file=os.path.join(self.data_dir, self.val_params["ann_file"]),
            transforms=make_transforms("val"),
            support_transforms=make_support_transforms(),
            return_masks=False,
            activated_class_ids=self.base_ids + self.novel_ids,
            with_support=False,
            cache_mode=self.cache_mode,
            local_rank=utils.get_local_rank(),
            local_size=utils.get_local_size(),
        )

        self.sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)

    def train_dataloader(self):
        """Train dataloader for the meta object detection."""

        return DataLoader(
            self.dataset_train,
            batch_sampler=self.batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Validate dataloader for the meta object detection."""

        return DataLoader(
            self.dataset_val,
            batch_size=self.val_params.batch_size,
            sampler=self.sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_support_dataloader(self):

        return DataLoader(
            self.dataset_support,
            batch_size=1,
            sampler=self.sampler_support,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
