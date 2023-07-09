# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods.cl_manager import CLManagerBase

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class ER(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)

    def update_memory(self, sample, sample_num=None):
        self.reservoir_memory(sample, sample_num)

    def reservoir_memory(self, sample, sample_num):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j, sample_num=sample_num)
        else:
            self.memory.replace_sample(sample, sample_num=sample_num)

    def report_training(self, sample_num, train_loss, train_acc):
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)
        self.writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc, cls_acc):
        print("cls_acc")
        print(cls_acc)
        self.writer.add_scalar(f"test/loss", avg_loss, sample_num)
        self.writer.add_scalar(f"test/acc", avg_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | TFLOPs {self.total_flops/1000:.2f}"
        )
        for idx in range(self.num_learned_class):
            acc = cls_acc[idx]
            logger.info(
                f"Class_Acc | Sample # {sample_num} | cls{idx} {acc:.4f}"
            )