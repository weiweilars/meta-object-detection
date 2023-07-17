#!/usr/bin/env python3
from typing import Any
import sys
import math
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from Meta_DETR.models import build_model
from Meta_DETR.util.lr_scheduler import WarmupMultiStepLR
from Meta_DETR.engine import sample_support_categories
from Meta_DETR.datasets.eval_detection import DetectionEvaluator
import Meta_DETR.util.misc as utils
import pdb


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def cal_category_code(
    support_dataloader,
    model,
    episode_size,
    device,
    num_feature_levels,
    postprocessors,
    base_ds,
    act_ids,
    number_of_supports=100,
):
    support_iter = iter(support_dataloader)
    all_category_codes_final = []
    print("Extracting support category codes...")
    for i in range(number_of_supports):
        support_iter = iter(support_dataloader)
        support_images, support_class_ids, support_targets = next(support_iter)
        support_images = [support_image.squeeze(0) for support_image in support_images]
        support_class_ids = support_class_ids.squeeze(0).to(device)
        support_targets = [
            {k: v.squeeze(0) for k, v in t.items()} for t in support_targets
        ]
        num_classes = support_class_ids.shape[0]
        num_episode = math.ceil(num_classes / episode_size)
        category_codes_final = []
        support_class_ids_final = []
        for i in range(num_episode):
            if (episode_size * (i + 1)) <= num_classes:
                support_images_ = utils.nested_tensor_from_tensor_list(
                    support_images[(episode_size * i) : (episode_size * (i + 1))]
                ).to(device)
                support_targets_ = [
                    {k: v.to(device) for k, v in t.items()}
                    for t in support_targets[
                        (episode_size * i) : (episode_size * (i + 1))
                    ]
                ]
                support_class_ids_ = support_class_ids[
                    (episode_size * i) : (episode_size * (i + 1))
                ]
            else:
                support_images_ = utils.nested_tensor_from_tensor_list(
                    support_images[-episode_size:]
                ).to(device)
                support_targets_ = [
                    {k: v.to(device) for k, v in t.items()}
                    for t in support_targets[-episode_size:]
                ]
                support_class_ids_ = support_class_ids[-episode_size:]

            category_code = model.compute_category_codes(
                support_images_, support_targets_
            )
            category_code = torch.stack(
                category_code, dim=0
            )  # (num_enc_layer, args.total_num_support, d)
            category_codes_final.append(category_code)
            support_class_ids_final.append(support_class_ids_)
        support_class_ids_final = torch.cat(support_class_ids_final, dim=0)
        category_codes_final = torch.cat(
            category_codes_final, dim=1
        )  # (num_enc_layer, num_episode x args.total_num_support, d)
        all_category_codes_final.append(category_codes_final)

    if num_feature_levels == 1:
        all_category_codes_final = torch.stack(
            all_category_codes_final, dim=0
        )  # (number_of_supports, num_enc_layer, num_episode x args.total_num_support, d)
        all_category_codes_final = torch.mean(
            all_category_codes_final, 0, keepdims=False
        )
        all_category_codes_final = list(torch.unbind(all_category_codes_final, dim=0))
    elif num_feature_levels == 4:
        raise NotImplementedError
    else:
        raise NotImplementedError
    print("Completed extracting category codes. Start Inference...")

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )

    iou_types = tuple(k for k in ("bbox",) if k in postprocessors.keys())
    evaluator = DetectionEvaluator(base_ds, iou_types)
    evaluator.coco_eval["bbox"].params.catIds = act_ids

    return support_class_ids_final, all_category_codes_final, evaluator


class MetaObjectModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(self, **kwargs):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.params = self.hparams.copy()
        # pdb.set_trace()
        #
        self.support_dataloader = self.params.pop("support_dataloader")
        self.val_base_ds = self.params.pop("val_base_ds")
        self.val_act_ids = self.params.pop("val_act_ids")

        self.model, self.criterion, self.postprocessors = build_model(self.params)

        self.model.to(self.params["device"])

        self.optimizer_params = self.params["optimizer"]

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(
        self,
        samples: torch.Tensor,
        targets: torch.Tensor,
        supp_samples: torch.Tensor,
        supp_class_ids: torch.Tensor,
        supp_targets: torch.Tensor,
    ):
        return self.model(samples, targets, supp_samples, supp_class_ids, supp_targets)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def training_step(self, batch: Any, batch_idx: int):
        samples, targets, support_images, support_class_ids, support_targets = batch

        (
            targets,
            support_images,
            support_class_ids,
            support_targets,
        ) = sample_support_categories(
            self.params["transformer"],
            targets,
            support_images,
            support_class_ids,
            support_targets,
        )

        samples = samples.to(self.params["device"])
        targets = [
            {k: v.to(self.params["device"]) for k, v in t.items()} for t in targets
        ]
        support_images = support_images.to(self.params["device"])
        support_class_ids = support_class_ids.to(self.params["device"])
        support_targets = [
            {k: v.to(self.params["device"]) for k, v in t.items()}
            for t in support_targets
        ]

        outputs = self(
            samples,
            targets=targets,
            supp_samples=support_images,
            supp_class_ids=support_class_ids,
            supp_targets=support_targets,
        )

        loss_dict = self.criterion(outputs)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(
                "Loss is NaN - {}. \nTraining terminated unexpectedly.\n".format(
                    loss_value
                )
            )
            print("loss dict:")
            print(loss_dict_reduced)
            sys.exit(1)

        # metric_logger.update(
        #     loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        # )
        # metric_logger.update(class_error=loss_dict_reduced["class_error"])
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(grad_norm=grad_total_norm)
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        # self.train_loss(loss)
        # self.train_acc(preds, targets)
        # self.log(
        #     "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        # )
        # self.log(
        #     "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        # )

        # # return loss or backpropagation will fail
        self.train_loss(losses)

        return losses

    def validation_step(self, batch: Any, batch_idx: int):

        # pdb.set_trace()
        samples, targets = batch
        samples = samples.to(self.params["device"])
        targets = [
            {k: v.to(self.params["device"]) for k, v in t.items()} for t in targets
        ]

        (
            support_class_ids_final,
            all_category_codes_final,
            self.evaluator,
        ) = cal_category_code(
            self.support_dataloader,
            self.model,
            self.params["transformer"]["episode_size"],
            self.params["device"],
            self.params["backbone"]["num_feature_levels"],
            self.postprocessors,
            self.val_base_ds,
            self.val_act_ids,
            number_of_supports=100,
        )

        outputs = self.model(
            samples,
            targets=targets,
            supp_class_ids=support_class_ids_final,
            category_codes=all_category_codes_final,
        )
        loss_dict = self.criterion(outputs)
        weight_dict = self.criterion.weight_dict

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        self.val_loss(losses)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return losses

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """

        if not self.params["share"]["fewshot_finetune"]:
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not match_name_keywords(
                            n, self.optimizer_params.lr_backbone_names
                        )
                        and not match_name_keywords(
                            n, self.optimizer_params.lr_linear_proj_names
                        )
                        and p.requires_grad
                    ],
                    "lr": self.optimizer_params.lr,
                    "initial_lr": self.optimizer_params.lr,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if match_name_keywords(
                            n, self.optimizer_params.lr_backbone_names
                        )
                        and p.requires_grad
                    ],
                    "lr": self.optimizer_params.lr_backbone,
                    "initial_lr": self.optimizer_params.lr_backbone,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if match_name_keywords(
                            n, self.optimizer_params.lr_linear_proj_names
                        )
                        and p.requires_grad
                    ],
                    "lr": self.optimizer_params.lr
                    * self.optimizer_params.lr_linear_proj_mult,
                    "initial_lr": self.optimizer_params.lr
                    * self.optimizer_params.lr_linear_proj_mult,
                },
            ]
        else:
            # For few-shot finetune stage, do not train sampling offsets, reference points, and embedding related parameters
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not match_name_keywords(
                            n, self.optimizer_params.lr_backbone_names
                        )
                        and not match_name_keywords(
                            n, self.optimizer_params.lr_linear_proj_names
                        )
                        and not match_name_keywords(
                            n, self.optimizer_params.embedding_related_names
                        )
                        and p.requires_grad
                    ],
                    "lr": self.optimizer_params.lr,
                    "initial_lr": self.optimizer_parms.lr,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if match_name_keywords(
                            n, self.optimizer_params.lr_backbone_names
                        )
                        and p.requires_grad
                    ],
                    "lr": self.optimizer_params.lr_backbone,
                    "initial_lr": self.optimizer_params.lr_backbone,
                },
            ]

        optimizer = torch.optim.AdamW(
            param_dicts, weight_decay=self.optimizer_params.weight_decay
        )
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            self.optimizer_params.lr_drop_milestones,
            gamma=0.1,
            warmup_epochs=self.optimizer_params.warmup_epochs,
            warmup_factor=self.optimizer_params.warmup_factor,
            warmup_method="linear",
            last_epoch=self.optimizer_params.start_epoch - 1,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
