import logging
import copy
import types
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time
import datetime

from methods.er_new import ER
from utils.data_loader import get_statistics, partial_distill_loss
from utils.train_utils import get_data_loader, select_model, select_optimizer, select_scheduler
from utils.afd import MultiTaskAFDAlternative
from utils.data_loader import cutmix_data
from utils.augmentations import CustomRandomCrop, CustomRandomHorizontalFlip, DoubleCompose, DoubleTransform
# from methods.cl_manager import MemoryBase

from utils.data_loader import MultiProcessLoader
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")
import math

class TWF(ER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        self.n_tasks = kwargs['n_tasks']
        self.samples_per_task = kwargs['samples_per_task']
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
        
        
        self.made_transform = transforms.Compose(
            [transforms.RandomCrop(224, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))]
                                  )
        self.buf_transform = self.get_custom_double_transform(self.made_transform.transforms)
        # Memory Switing
        # self.memory = TwFMemory(self.memory_size)
        self.initialize()
        
    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, \
            self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.future_task_id = 0
        self.future_samples_cnt = 0
        self.memory = TwFMemory(self.memory_size)
       
        #self.memory = MemoryBase(self.memory_size)
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
        self.temp_sample_nums = []

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        
        for i in range(self.future_steps):
            self.load_batch()
    
    def get_batch(self):
        batch = self.dataloader.get_batch()
        self.load_batch()
        return batch
    
    
    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch)
            self.waiting_batch_idx.append(self.temp_future_batch_idx)
            
            
    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
            sample['task_id'] = self.future_task_id
            self.future_samples_cnt += 1
            self.future_task_id = self.future_samples_cnt // self.samples_per_task
        except:
            # 더 이상 sample이 남아있지 않으면 종료함
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
        self.temp_future_batch.append(sample)
        self.temp_future_batch_idx.append(self.future_sample_num)
        self.future_num_updates += self.online_iter

        if len(self.temp_future_batch) >= self.temp_batch_size:
            self.generate_waiting_batch(int(self.future_num_updates))
            # for idx, stored_sample in enumerate(self.temp_future_batch):
            #     self.update_memory(stored_sample, stored_sample['task_id'], self.future_samples_cnt -(self.temp_batch_size) + idx)
            self.temp_future_batch = []
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0
    
    
    def initialize(self):
        
        # Teacher model
        self.model = select_model(self.model_name, self.dataset, pre_trained=True).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes).to(self.device)
        
        self.pretrained_model = copy.deepcopy(self.model)
        self.pretrained_model = self.pretrained_model.to(self.device)
        self.pretrained_model = self.pretrained_model.eval()
        
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
        
        self.model.set_return_prerelu(True)
        self.pretrained_model.set_return_prerelu(True)
        
    def online_before_task(self, task_id):
        self.task_id = task_id
        if task_id == 0:
            x = torch.zeros((self.temp_batch_size, 3, self.inp_size, self.inp_size))
        
            x = x.to(self.device)
            _, feats_t = self.model(x, get_features=True)
            _, pret_feats_t = self.pretrained_model(x, get_features=True)
            
            for i, (x, pret_x) in enumerate(zip(feats_t, pret_feats_t)):
                # clear_grad=self.args.detach_skip_grad == 1
                adapt_shape = x.shape[1:]
                pret_shape = pret_x.shape[1:]
                if len(adapt_shape) == 1:
                    adapt_shape = (adapt_shape[0], 1, 1)  # linear is a cx1x1
                    pret_shape = (pret_shape[0], 1, 1)
                
                setattr(self.model, f"adapter_{i+1}", MultiTaskAFDAlternative(
                    adapt_shape, self.n_tasks, clear_grad=False,
                    teacher_forcing_or=False,
                    lambda_forcing_loss=self.lambda_fp_replay,
                    use_overhaul_fd=True, use_hard_softmax=True,
                    lambda_diverse_loss=self.lambda_diverse_loss,
                    attn_mode="chsp",
                    min_resize_threshold=self.min_resize_threshold,
                    resize_maps=self.resize_maps == 1,
                ).to(self.device))

    def online_after_task(self):
        self.model.eval()
        self.memory.loop_over_buffer(self.model, self.pretrained_model, self.batch_size, self.future_task_id, self.device)
        
    
    def online_step(self, sample, sample_num, n_worker):
        self.sample_num  = sample_num
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_sample_nums.append(sample_num)
            
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            
            if int(self.num_updates) > 0:
                images, train_loss, train_acc, logits, attention_maps, labels = self.online_train(iterations=int(self.num_updates))
                self.report_training(sample_num, train_loss, train_acc)
                # for idx, stored_sample in enumerate(self.temp_batch):
                #     self.complete_memory(images[idx], self.temp_sample_nums[idx], logits[idx],[at_map[idx] for at_map in attention_maps], labels[idx])
                    
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []
            self.temp_sample_nums = []
    
    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)

        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})

    
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

        #self.total_flops += (len(y) * self.forward_flops)

        if get_feature:
            return logit, loss, feature
        elif get_features:
            return logit, loss, features
        else:
            return logit, loss
    
    
    def get_custom_double_transform(self, transform):
        tfs = []
        for tf in transform:
            if isinstance(tf, transforms.RandomCrop):
                tfs.append(CustomRandomCrop(tf.size, tf.padding, resize=False, min_resize_index=2))
            elif isinstance(tf, transforms.RandomHorizontalFlip):
                tfs.append(CustomRandomHorizontalFlip(tf.p))
            elif isinstance(tf, transforms.Compose):
                tfs.append(DoubleCompose(
                    self.get_custom_double_transform(tf.transforms)))
            else:
                tfs.append(DoubleTransform(tf))

        return DoubleCompose(tfs)
    
    def online_train(self, iterations=1):
        
        self.model.train()
        self.pretrained_model.eval()
        
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.online_train_ct += 1
        for i in range(iterations):
            data = self.get_batch()
            stream_x = data["image"].to(self.device)
            stream_y = data["label"].to(self.device)
            stream_task_ids = data['task_id'].to(self.device)
            memory_x, memory_y, memory_task_ids = None, None, None
            # self.optimizer.zero_grad()
        
            # SET variables => stream, memory, all (stream + memory)
            # e.g. stream_x, memory_x, x (all)
            if len(self.memory.objects) > 0:
                memory_data, _ = self.memory.retrieval(self.temp_batch_size)
                memory_x = [item['image'] for item in memory_data]
                memory_attentionmaps = [[attention_map.to(self.device) for attention_map in item['attention_maps']] for item in memory_data]
                d = [self.buf_transform(ee.float(), attn_map) for ee, attn_map in zip(memory_x, memory_attentionmaps)]
                memory_attentionmaps = [v[1] for v in d]
                memory_x = [v[0] for v in d]
                memory_x = torch.stack([memory_image.to(self.device) for memory_image in memory_x])
                memory_y = torch.tensor([item['label'] for item in memory_data]).to(self.device)
                memory_task_ids = [item['task_id'] for item in memory_data]
                x = torch.stack(list(stream_x) + list(memory_x)).to(self.device)
                y = torch.stack(list(stream_y) + list(memory_y)).to(self.device)
                task_ids = torch.tensor(list(stream_task_ids) + list(memory_task_ids)).to(self.device)
            else:
                x = stream_x.to(self.device)
                y = stream_y.to(self.device)
                task_ids = torch.tensor(stream_task_ids).to(self.device)
                
            
            logit, loss, all_features = self.model_forward(x, y, get_features=True)
            all_pret_logits, all_pret_features = self.pretrained_model(x, get_features=True)
            
            all_features = all_features[:-1]
            all_pret_features = all_pret_features[:-1]
            
            self.optimizer.zero_grad()
            
            stream_logit = logit[:self.temp_batch_size]
            memory_logit = logit[self.temp_batch_size:]
            
            stream_partial_features = [feature[:self.temp_batch_size] for feature in all_features]
    
            stream_pret_logits = all_pret_logits[:self.temp_batch_size]
            stream_pret_partial_features = [feature[:self.temp_batch_size] for feature in all_pret_features]
            
            loss_er = torch.tensor(0.)
            loss_der = torch.tensor(0.)
            loss_afd = torch.tensor(0.)
            # buf_data = self.memory.buffer
            # buf_sample_ids = [d['sample_id'] for d in buf_data]
            
            if len(self.memory.objects) == 0:
                loss_afd, stream_attention_maps = partial_distill_loss(self.model, 
                        stream_partial_features, stream_pret_partial_features, stream_y, stream_task_ids, self.device)
            else:
                
                
                memory_logits = [item['logits'].to(self.device) for item in memory_data]
                memory_labels = [item['label'] for item in memory_data]
    
                buffer_teacher_forcing = [task_ids != self.future_task_id for task_ids in memory_task_ids]
                buffer_teacher_forcing = torch.tensor(buffer_teacher_forcing).to(self.device)
                teacher_forcing = torch.cat(
                    (torch.zeros((self.temp_batch_size)).bool().to(self.device), buffer_teacher_forcing))
                attention_maps = [
                    [torch.ones_like(map) for map in memory_attentionmaps[0]]]*self.temp_batch_size + memory_attentionmaps
                
                
                loss_afd, all_attention_maps = partial_distill_loss(self.model, all_features, all_pret_features, y, task_ids, self.device,
                    teacher_forcing, attention_maps)
                loss_afd.to(self.device)

                stream_attention_maps = [ap[:self.temp_batch_size] for ap in all_attention_maps]
                memory_classes = [item['cls'] for item in memory_data]
                memory_logits = torch.stack(memory_logits)
                memory_labels = torch.stack(memory_labels).to(self.device)
                
                loss_er = self.loss(memory_logit, memory_y)
            #     loss_der = F.mse_loss(buf_logit, memory_logits)
                
                for i in range(len(memory_logits)):
                    length = memory_classes[i]
                    # resize to make equal # of class conditions
                    buf_logit_seg = memory_logit[i][:length]
                    memory_logit_seg = memory_logits[i][:length]
                    
                    # substract mean
                    buf_logit_seg = (buf_logit_seg - torch.mean(buf_logit_seg, dim=0))
                    memory_logit_seg = (memory_logit_seg - torch.mean(memory_logit_seg, dim=0))
                    
                    # add per logit
                    loss_der += F.mse_loss(buf_logit_seg, memory_logit_seg).cpu()
            # try:
            #     # print(stream_attention_maps)
            # except:
            #     print("no memory attention maps")
            # stream_y = torch.tensor(y[:self.temp_batch_size]).clone().detach()
            stream_y = y[:self.temp_batch_size]
            loss = self.loss(stream_logit, stream_y)
            loss += self.kwargs['der_beta'] * loss_er
            loss += self.kwargs['der_alpha'] * loss_der
            loss += self.kwargs['lambda_fp'] * loss_afd

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

        # return total_loss / iterations, correct / num_data, logit, stream_attention_maps
        stream_logit = [logit.detach().cpu()  for logit in stream_logit]
        stream_attention_maps = [attention_map.detach().cpu() for attention_map in stream_attention_maps]
        
        x = x.detach().cpu()
        y = y.detach().cpu()
        
        stream_attention_maps_iter = []
        for idx in range(self.temp_batch_size):
            stream_attention_maps_iter.append([at_map[idx] for at_map in stream_attention_maps])
        
        self.update_complete_memory(self.train_datalist[(self.online_train_ct * self.temp_batch_size - self.temp_batch_size) : self.online_train_ct * self.temp_batch_size], \
            self.task_id, x[:self.temp_batch_size], stream_logit, stream_attention_maps_iter, y[:self.temp_batch_size])
        self.future_sample_num += 1
        
        return x[:self.temp_batch_size], total_loss / iterations, correct / num_data, stream_logit, stream_attention_maps, y[:self.temp_batch_size]

    # # Future에서 뽑은 데이터를 memory buffer에 저장 (Complete 전)
    # def update_memory(self, sample, task_id, sample_id):
    #     self.memory.select_sample(sample, task_id, sample_id)
        
    # # Memory buffer에 저장되어 있는 데이터를 Replay memory로 옮김 (Complete)
    # def complete_memory(self, image, sample_id, logits, attention_maps, labels):
    #     self.memory.replace_sample(image, sample_id, logits, attention_maps, labels)

    # 저 위에 있는 2개를 한 번에 함
    def update_complete_memory(self, sample, task_id, image, logits, attention_maps, labels):
        self.memory.replace_through(sample, task_id, image, logits, attention_maps, labels, sample_id=self.future_sample_num)
        
    def report_training(self, sample_num, train_loss, train_acc):
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)
        self.writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc, cls_acc):
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
# data --(select_sample)--> buffer[object] --(replace_sample)--> replay memory[object]
# object keys: sample, label, sample_id, task_id, logits, attention_maps
class TwFMemory():
    def __init__(self, memory_size):
        self.seen = 0
        self.memory_size = memory_size
        self.buffer = [] # without attention map and logits, you should complete it
        self.objects  = []
        self.cls_list = []
        self.cls_dict = dict()
        self.cls_count = []
        self.cls_idx = []
        self.usage_count = np.array([])
        self.class_usage_count = np.array([])
        self.current_images = []
        self.current_labels = []
        self.current_cls_count = [0 for _ in self.cls_list]
        self.current_cls_idx = [[] for _ in self.cls_list]
        
    def __len__(self):
        return len(self.objects)
    
    def select_sample(self, sample, task_id, sample_id):
        self.buffer.append(self.create_data(sample, task_id, sample_id))
    
    def replace_sample(self, image, sample_id, logits, attention_maps, labels):
        self.seen += 1
        for logit in logits:
            logit = logit.detach()
        for attention_map in attention_maps:
            attention_map = attention_map.detach()
        
        for buffer_idx, item in enumerate(self.buffer):
            if sample_id == item['sample_id']:
                item['logits'] = logits
                item['attention_maps'] = attention_maps
                item['image'] = image
                item['label'] = labels
                
                if len(self.objects) >= self.memory_size:
                    idx = np.random.randint(self.seen)
                    if idx < len(self.objects):
                        self.objects[idx] = item
                else:
                    self.objects.append(item)
                    
                del self.buffer[buffer_idx]
                break
    
    def replace_through(self, sample, task_id, image, logits, attention_maps, labels, sample_id):
        
        for i in range(len(sample)):
            self.seen += 1
            if len(self.objects) >= self.memory_size:
                idx = np.random.randint(self.seen)
                if idx < len(self.objects):
                    self.objects[idx] = self.create_data(sample[i], task_id, None, image[i], logits[i], attention_maps[i], labels[i])
            else:
                self.objects.append(self.create_data(sample[i], task_id, None, image[i], logits[i], attention_maps[i], labels[i]))
        
        
    def whole_retrieval(self):
        memory_batch = []
        indices = list(range(len(self.objects)))
        for i in indices:
            memory_batch.append(self.objects[i])
        return memory_batch, indices
    
    def retrieval(self, size):
        sample_size = min(size, len(self.objects))
        memory_batch = []
        indices = np.random.choice(range(len(self.objects)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.objects[i])
        return memory_batch, indices
    
    def create_data(self, sample, task_id, sample_id, image=None, logits=None, attention_maps=None, labels=None):
        data = {}
        data['sample'] = sample
        data['label'] = self.cls_dict[sample['klass']]
        data['klass'] = sample['klass']
        data['task_id'] = task_id
        data['sample_id'] = sample_id
        data['image'] = image
        data['logits'] = logits
        data['attention_maps'] = attention_maps
        data['cls'] = len(self.cls_list)
        data['label'] = labels
        
        return data


    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.cls_list)
        self.cls_list.append(class_name)
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.class_usage_count = np.append(self.class_usage_count, 0.0)
        print("!!added", class_name)
        print("self.cls_dict", self.cls_dict)

    def batch_iterate(self, size, batch_size):
        n_chunks = size // batch_size
        for i in range(n_chunks):
            yield torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size)))

    def loop_over_buffer(self, model, pretrained_model, batch_size, task_id, device):

        with torch.no_grad():
            # loop over memory
            for obj_idxs in self.batch_iterate(len(self.objects), batch_size):

                obj_idxs = torch.tensor(obj_idxs.to(device))
                obj_task_ids = torch.tensor([self.objects[idx]['task_id'] for idx in obj_idxs]).to(device)
                obj_labels = torch.tensor([self.objects[idx]['label'] for idx in obj_idxs]).to(device)

                obj_mask = [obj_task_id == task_id for obj_task_id in obj_task_ids]
                obj_mask = torch.tensor(obj_mask)

                if not obj_mask.any():
                    continue
                
                obj_inputs = [self.objects[idx]['image'] for idx in obj_idxs]
                obj_inputs = torch.stack(obj_inputs).to(device)
                obj_labels = torch.stack(obj_labels).to(device)
                obj_inputs = obj_inputs[obj_mask]
                obj_labels = obj_labels[obj_mask]
                obj_idxs = obj_idxs[obj_mask]

                # buf_inputs = torch.stack([self.not_aug_transform(
                #     ee.cpu()) for ee in buf_inputs]).to(self.device)

                _, _, mem_partial_features = model(
                    obj_inputs, get_features=True)
                _, pret_mem_partial_features = pretrained_model(obj_inputs, get_features=True)

                _, attention_masks = partial_distill_loss(mem_partial_features, pret_mem_partial_features, obj_labels, obj_task_ids, device)

                for idx in obj_idxs:
                    self.objects[idx]['attention_maps'] = [
                        at[idx % len(at)] for at in attention_masks]