from detectron2.data.datasets import register_coco_instances
import copy
import itertools
import logging
import os
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Set
import detectron2.utils.comm as comm
import torch

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.solver.build import maybe_add_gradient_clipping

from Mask2Former.mask2former import MaskFormerSemanticDatasetMapper, \
    MaskFormerPanopticDatasetMapper, MaskFormerInstanceDatasetMapper, \
    COCOInstanceNewBaselineDatasetMapper, COCOPanopticNewBaselineDatasetMapper, \
    InstanceSegEvaluator, add_maskformer2_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, \
    MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
)
from detectron2.evaluation import print_csv_format, SemSegEvaluator, COCOEvaluator, \
    COCOPanopticEvaluator, CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, \
    LVISEvaluator, DatasetEvaluators
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

import pytorch_lightning as pl  # type: ignore
from pytorch_lightning import LightningDataModule, LightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")


class TrainingModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.storage: EventStorage = None
        self.model = build_model(self.cfg)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["iteration"] = self.storage.iter

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        self.start_iter = checkpointed_state["iteration"]
        self.storage.iter = self.start_iter

    def setup(self, stage: str):
        if self.cfg.MODEL.WEIGHTS:
            self.checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.model,
                self.cfg.OUTPUT_DIR,
            )
            logger.info(
                f"Load model weights from checkpoint: {self.cfg.MODEL.WEIGHTS}.")
            # Only load weights, use lightning checkpointing if you want to resume
            self.checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.iteration_timer = hooks.IterationTimer()
        self.iteration_timer.before_train()
        self.data_start = time.perf_counter()
        self.writers = None

    def training_step(self, batch, batch_idx):
        data_time = time.perf_counter() - self.data_start
        # Need to manually enter/exit since trainer may launch processes
        # This ideally belongs in setup, but setup seems to run before processes are spawned
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()
            self.iteration_timer.trainer = weakref.proxy(self)
            self.iteration_timer.before_step()
            self.writers = (
                default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
                if comm.is_main_process()
                else {}
            )

        loss_dict = self.model(batch)
        SimpleTrainer.write_metrics(loss_dict, data_time)

        opt = self.optimizers()
        self.storage.put_scalar(
            "lr", opt.param_groups[self._best_param_group_id]["lr"],
            smoothing_hint=False
        )
        self.iteration_timer.after_step()
        self.storage.step()
        # A little odd to put before step here, but it's the best way to get a proper timing
        self.iteration_timer.before_step()

        if self.storage.iter % 20 == 0:
            for writer in self.writers:
                writer.write()
        return sum(loss_dict.values())

    def training_step_end(self, training_step_outpus):
        self.data_start = time.perf_counter()
        return training_step_outpus

    def training_epoch_end(self, training_step_outputs):
        self.iteration_timer.after_train()
        if comm.is_main_process():
            self.checkpointer.save("model_final")
        for writer in self.writers:
            writer.write()
            writer.close()
        self.storage.__exit__(None, None, None)

    def _process_dataset_evaluation_results(self) -> OrderedDict:
        results = OrderedDict()
        for idx, dataset_name in enumerate(self.cfg.DATASETS.TEST):
            results[dataset_name] = self._evaluators[idx].evaluate()
            if comm.is_main_process():
                print_csv_format(results[dataset_name])

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def _reset_dataset_evaluators(self):
        self._evaluators = []
        for dataset_name in self.cfg.DATASETS.TEST:
            evaluator = self.build_evaluator(self.cfg, dataset_name)
            evaluator.reset()
            self._evaluators.append(evaluator)

    def on_validation_epoch_start(self, _outputs):
        self._reset_dataset_evaluators()

    def validation_epoch_end(self, _outputs):
        results = self._process_dataset_evaluation_results(_outputs)

        flattened_results = flatten_results_dict(results)
        for k, v in flattened_results.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    "[EvalHook] eval_function should return a nested dict of float. "
                    "Got '{}: {}' instead.".format(k, v)
                ) from e
        self.storage.put_scalars(**flattened_results, smoothing_hint=False)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch]
        outputs = self.model(batch)
        self._evaluators[dataloader_idx].process(batch, outputs)

    def configure_optimizers(self):
        optimizer = self.build_optimizer(self.cfg, self.model)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams[
                                            "lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(
                    COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True,
                                                  output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True,
                                                  output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                        torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                        torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    def train_dataloader(self):
        return self.build_train_loader(self.cfg)

    def val_dataloader(self):
        dataloaders = []
        for dataset_name in self.cfg.DATASETS.TEST:
            dataloaders.append(self.build_evaluator(self.cfg, dataset_name))
        return dataloaders


def main(args):
    cfg = setup(args)
    train(cfg, args)


def train(cfg, args):
    trainer_params = {
        # training loop is bounded by max steps, use a large max_epochs to make
        # sure max_steps is met first
        "max_epochs": 10 ** 8,
        "max_steps": cfg.SOLVER.MAX_ITER,
        "val_check_interval": cfg.TEST.EVAL_PERIOD if cfg.TEST.EVAL_PERIOD > 0 else 10 ** 8,
        "num_nodes": args.num_machines,
        "accelerator": "gpu",
        "devices": args.num_gpus,
        "num_sanity_val_steps": 0,
        "strategy": args.distributed_strategy,
    }
    if cfg.SOLVER.AMP.ENABLED:
        trainer_params["precision"] = 16
    print(cfg.DATASETS)
    register_coco_instances(cfg.DATASETS.TRAIN[0], {},
                            cfg.DATASETS.TRAIN[1],
                            cfg.DATASETS.TRAIN[2])
    register_coco_instances(cfg.DATASETS.TEST[0], {},
                            cfg.DATASETS.TEST[1],
                            cfg.DATASETS.TEST[2])
    cfg.defrost()
    cfg.DATASETS.TRAIN = ('train_dataset',)
    cfg.DATASETS.TEST = ('test_dataset',)
    cfg.freeze()

    last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "last.ckpt")
    if args.resume:
        # resume training from checkpoint
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")

    trainer = pl.Trainer(**trainer_params)
    logger.info(
        f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")

    module = TrainingModule(cfg)
    data_module = DataModule(cfg)
    if args.eval_only:
        logger.info("Running inference")
        trainer.validate(module, data_module)
    else:
        logger.info("Running training")
        trainer.fit(module, data_module)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(),
                 name="mask2former")
    return cfg


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--distributed_strategy",type=str,default="ddp_sharded")
    args = parser.parse_args()
    logger.info("Command Line Args:", args)
    print("Command Line Args:", args)
    main(args)
