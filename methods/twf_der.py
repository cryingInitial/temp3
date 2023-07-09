import logging
import copy
import types
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods.er_new import ER
from utils.data_loader import get_statistics, partial_distill_loss
from utils.train_utils import get_data_loader, select_model
from utils.afd import MultiTaskAFDAlternative
from utils.data_loader import cutmix_data
# from methods.cl_manager import MemoryBase
from methods.cl_manager import CLManagerBase, MemoryBase
from utils.data_loader import MultiProcessLoader
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")
import math


class TWF_DER(ER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        self.n_tasks = kwargs['n_tasks']
        self.samples_per_task = kwargs['samples_per_task']
        self.writer = SummaryWriter(f'tensorboard/{kwargs["dataset"]}/{kwargs["note"]}/seed_{kwargs["rnd_seed"]}')
        self.sub_future_queue = deque()
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
        self.n_classes = n_classes
        self.inp_size = inp_size
        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.kwargs = kwargs
        self.lambda_fp_replay = kwargs['lambda_fp_replay']
        self.lambda_diverse_loss = kwargs['lambda_diverse_loss']
        self.min_resize_threshold = 16
        self.resize_maps = 0
        self.online_train_ct = 0
        self.batch_size = kwargs['batchsize']
            
        self.initialize()
    def initialize(self):
        # Teacher model
        # self.model = select_model(self.model_name, self.dataset, 1, pre_trained=True).to(self.device)
        # self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes).to(self.device)
        # self.pretrained_model = select_model(self.model_name, self.dataset, 1, pre_trained=True).to(self.device).eval()
       
        # for p in self.pretrained_model.parameters():
            # p.requires_grad = False
        
        # self.model.set_return_prerelu(True)
        # self.pretrained_model.set_return_prerelu(True)
        pass
    def online_before_task(self, task_id):
        self.task_id = task_id
        # if task_id == 0:
        #     x = torch.zeros((self.temp_batch_size, 3, self.inp_size, self.inp_size))
        
        #     x = x.to(self.device)
        #     _, feats_t = self.model(x, get_features=True)
        #     _, pret_feats_t = self.pretrained_model(x, get_features=True)
            
        #     for i, (x, pret_x) in enumerate(zip(feats_t, pret_feats_t)):
        #         # clear_grad=self.args.detach_skip_grad == 1
        #         adapt_shape = x.shape[1:]
        #         pret_shape = pret_x.shape[1:]
        #         if len(adapt_shape) == 1:
        #             adapt_shape = (adapt_shape[0], 1, 1)  # linear is a cx1x1
        #             pret_shape = (pret_shape[0], 1, 1)
                
        #         setattr(self.model, f"adapter_{i+1}", MultiTaskAFDAlternative(
        #             adapt_shape, self.n_tasks, clear_grad=False,
        #             teacher_forcing_or=False,
        #             lambda_forcing_loss=self.lambda_fp_replay,
        #             use_overhaul_fd=True, use_hard_softmax=True,
        #             lambda_diverse_loss=self.lambda_diverse_loss,
        #             attn_mode="chsp",
        #             min_resize_threshold=self.min_resize_threshold,
        #             resize_maps=self.resize_maps == 1,
        #         ).to(self.device))
                
    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = TWFMemory(self.memory_size, self.device)

        self.logit_num_to_get = []
        self.logit_num_to_save = []

        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True
        self.task_id = 0
        self.logit_num_to_get = []
        self.logit_num_to_save = []
        self.waiting_batch = []
        for i in range(self.future_steps):
            self.load_batch()
            
    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
        self.temp_future_batch.append(sample)
        self.temp_future_batch_idx.append(self.future_sample_num)
        self.future_num_updates += self.online_iter
        
        if len(self.temp_future_batch) >= self.temp_batch_size:
            self.generate_waiting_batch(int(self.future_num_updates))
            temp_batch_logit_num = []
            for future_sample_num, stored_sample in zip(self.temp_future_batch_idx, self.temp_future_batch):
                # 이거 task_id 나중에 고쳐야 함
                logit_num = self.update_memory(sample, self.task_id, future_sample_num)
                temp_batch_logit_num.append(logit_num)
            self.logit_num_to_save.append(temp_batch_logit_num)
            self.temp_future_batch = []
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0
    
    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            memory_batch, logit_nums, memory_batch_idx = self.memory.retrieval(self.memory_batch_size)
            self.waiting_batch.append(self.temp_future_batch + memory_batch)
            self.logit_num_to_get.append(logit_nums)
            self.waiting_batch_idx.append(self.temp_future_batch_idx + memory_batch_idx)
    
    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            train_loss, train_acc, logits, atten_maps = self.online_train(iterations=int(self.num_updates))
            for i, num in enumerate(self.logit_num_to_save[0]):
                if num is not None:
                    self.memory.save_logits(num, logits[i], atten_maps[i])
            del self.logit_num_to_save[0]
            self.report_training(sample_num, train_loss, train_acc)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)


    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            print(y)
            sample_nums = data["sample_nums"].to(self.device)
            task_ids = data['task_id']
            task_ids[:self.temp_batch_size] = torch.LongTensor([self.task_id] * self.temp_batch_size)            
            
            # logit, loss, features = self.model_forward(x, y, get_features=True)
            logit, loss = self.model_forward(x, y)
            # pret_logit, pret_features = self.pretrained_model(x, get_features=True)
            
            stream_logit = logit[:self.temp_batch_size]
            # stream_pret_logit = pret_logit[:self.temp_batch_size]
            
            # stream_features = [feature[:self.temp_batch_size] for feature in features]
            # stream_pret_features = [feature[:self.temp_batch_size] for feature in pret_features]
            
            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            
        return total_loss / iterations, correct / num_data, [torch.zeros(50)] * self.temp_batch_size, [torch.zeros(1)] * self.temp_batch_size
    
    def update_memory(self, sample, tid, sample_num):
        logit_num = self.reservoir_memory(sample, tid, sample_num)
        return logit_num

    def reservoir_memory(self, sample, tid, sample_num):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                logit_num = self.memory.replace_sample(sample, tid, j, sample_num = sample_num)
            else:
                logit_num = None
        else:
            logit_num = self.memory.replace_sample(sample, tid, sample_num = sample_num)
        return logit_num
    
    def model_forward(self, x, y, get_feature=False, get_features=False):
        #do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        do_cutmix = False
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                if get_feature:
                    logit, feature = self.model(x, get_feature=True)
                else:
                    logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                if get_feature:
                    logit, feature = self.model(x, get_feature=True)
                elif get_features:
                    logit, features = self.model(x, get_features=True)
                else:
                    logit = self.model(x)
                loss = self.criterion(logit, y)
                
        if get_feature:
            return logit, loss, feature
        elif get_features:
            return logit, loss, features
        else:
            return logit, loss
class TWFMemory(MemoryBase):
    def __init__(self, memory_size, device):
        super().__init__(memory_size, device)
        self.tids = []
        self.logits = []
        self.logit_num = []
        self.atten_maps = []
        
    def replace_sample(self, sample, tid, idx=None, sample_num=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        sample['task_id'] = tid
        logit_num = len(self.logits)
        if idx is None:
            assert len(self.images) < self.memory_size
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
            self.tids.append(tid)
            self.logit_num.append(logit_num)
            self.logits.append(None)
            self.atten_maps.append(None)
            if sample_num is not None:
                self.sample_nums.append(sample_num)
        else:
            assert idx < self.memory_size
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.images[idx] = sample
            self.labels[idx] = self.cls_dict[sample['klass']]
            self.tids[idx] = tid
            self.logit_num[idx] = logit_num
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            self.logits.append(None)
            self.atten_maps.append(None)
            if sample_num is not None:
                self.sample_nums[idx] = sample_num
        return logit_num
            
    def retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        memory_batch_idx = []
        batch_logit_num = []
        indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.images[i])
            batch_logit_num.append(self.logit_num[i])
            memory_batch_idx.append(self.sample_nums[i])
        return memory_batch, batch_logit_num, memory_batch_idx
        
    def get_logit(self, logit_nums, num_classes):
        logits = []
        logit_masks = []
        atten_maps = []
        for i in logit_nums:
            len_logit = len(self.logits[i])
            logits.append(torch.cat([self.logits[i], torch.zeros(num_classes-len_logit)]))
            logit_masks.append(torch.cat([torch.ones(len_logit), torch.zeros(num_classes-len_logit)]))
            atten_maps.append(self.atten_maps[i])
        return torch.stack(logits), torch.stack(logit_masks), torch.stack(atten_maps)
    
    def save_logits(self, logit_num, logit, atten_map):
        self.logits[logit_num] = logit
        self.atten_maps[logit_num] = atten_map