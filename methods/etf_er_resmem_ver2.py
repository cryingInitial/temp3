# When we make a new one, we should inherit the Finetune class.
import logging
import numpy as np
from methods.cl_manager import CLManagerBase
from utils.train_utils import DR_loss, Accuracy
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
try:
    # version < 3.8
    import pickle5 as pickle
except ImportError:
    # version >= 3.8
    import pickle
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader, get_statistics
import math
import utils.train_utils 
import os
from utils.data_worker import load_data
from utils.train_utils import select_optimizer, select_model, select_scheduler
logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

class ETF_ER_RESMEM_VER2(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.in_channels = self.model.fc.in_features
        # 아래 것들 config 파일에 추가!
        # num class = 100, eval_class = 60 
        self.num_classes = kwargs["num_class"]
        self.eval_classes = 0 #kwargs["num_eval_class"]
        self.cls_feature_length = 50
        self.feature_mean_dict = {}
        self.current_cls_feature_dict = {}
        self.feature_std_mean_list = []
        self.current_feature_num = kwargs["current_feature_num"]
        self.residual_num = kwargs["residual_num"]
        self.distill_beta = kwargs["distill_beta"]
        self.distill_strategy = kwargs["distill_strategy"]
        self.distill_threshold = kwargs["distill_threshold"]

        self.stds_list = []
        self.softmax = nn.Softmax(dim=0).to(self.device)
        self.use_feature_distillation = kwargs["use_feature_distillation"]
        if self.loss_criterion == "DR":
            if self.use_feature_distillation:
                self.criterion = DR_loss(reduction="none").to(self.device)
            else:
                self.criterion = DR_loss().to(self.device)
        elif self.loss_criterion == "CE":
            if self.use_feature_distillation:
                self.criterion = nn.CrossEntropyLoss(reduction="none").to(self.device)
            else:
                self.criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        self.compute_accuracy = Accuracy(topk=self.topk)
        self.model = select_model(self.model_name, self.dataset, 1, pre_trained=False).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        self.etf_initialize()
        self.residual_dict = {}
        self.feature_dict = {}
        self.cls_feature_dict = {}
        self.note = kwargs["note"]
        os.makedirs(f"{self.note}", exist_ok=True)

    def get_cos_sim(self, a, b):
        inner_product = (a * b).sum(dim=1)
        a_norm = a.pow(2).sum(dim=1).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (a_norm * b_norm)
        return cos

    def sample_inference(self, samples):
        with torch.no_grad():
            self.model.eval()
            batch_labels = []
            batch_feature_dict = {}
            for sample in samples:
                x = load_data(sample, self.data_dir, self.test_transform).unsqueeze(0)
                y = self.cls_dict[sample['klass']]
                batch_labels.append(y)
                x = x.to(self.device)
                _, sample_feature = self.model(x, get_feature=True)

                if y not in batch_feature_dict.keys():
                    batch_feature_dict[y] = [sample_feature]
                else:
                    batch_feature_dict[y].append(sample_feature)

            for y in list(set(batch_labels)):
                if y not in self.cls_feature_dict.keys():
                    self.cls_feature_dict[y] = torch.mean(torch.stack(batch_feature_dict[y]), dim=0)
                else:
                    self.cls_feature_dict[y] = self.distill_coeff * self.cls_feature_dict[y] + (1-self.distill_coeff) * torch.mean(torch.stack(batch_feature_dict[y]), dim=0)


    def model_forward(self, x, y):
        #do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        do_cutmix = False

        """Forward training data."""
        target = self.etf_vec[:, y].t()
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit, feature = self.model(x, get_feature=True)
                #loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                loss = lam * self.criterion(feature, labels_a) + (1 - lam) * self.criterion(feature, labels_b)
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit, feature = self.model(x, get_feature=True)
                feature = self.pre_logits(feature)

                # calcualte current feature
                for idx, y_i in enumerate(y):
                    if y_i not in self.current_cls_feature_dict.keys():
                        self.current_cls_feature_dict[y_i.item()] = [feature[idx].detach().cpu()]
                    else:
                        self.current_cls_feature_dict[y_i.item()].append(feature[idx].detach().cpu())

                # check over storing
                for y_i in torch.unique(y):
                    if len(self.current_cls_feature_dict[y_i.item()]) >= self.current_feature_num:
                        self.current_cls_feature_dict[y_i.item()] = self.current_cls_feature_dict[y_i][-self.current_cls_feature_num:]

                if self.loss_criterion == "DR":
                    loss = self.criterion(feature, target)
                    residual = (target - feature).detach().cpu()
                elif self.loss_criterion == "CE":
                    loss = self.criterion(logit, y)
                    residual = (F.one_hot(y, num_classes=self.num_learned_class) - self.softmax(logit/self.softmax_temperature)).detach().cpu()
                elif self.loss_criterion == "DR_ANGLE":
                    loss = self.criterion(logit, y)
                    cos_theta = (target * feature) / torch.norm(target * feature, p=2, dim=1, keepdim=True)

                if self.use_feature_distillation:
                    past_feature = torch.stack([self.pre_logits(self.cls_feature_dict[y_i.item()]) for y_i in y], dim=0)
                    target_fc = torch.stack([self.etf_vec[:, y_i.item()] for y_i in y], dim=0)
                    l2_loss = ((feature - past_feature.detach().squeeze()) ** 2).sum(dim=1)
                    # naive distllation
                    if self.distill_strategy == "naive":
                        print("loss", loss.mean(), "l2_loss", self.distill_beta * l2_loss.mean())
                        loss = (loss + self.distill_beta * l2_loss).mean()

                    # similarity based distillation
                    elif self.distill_strategy == "classwise":
                        past_cos_sim = torch.abs(self.get_cos_sim(target_fc, past_feature.detach().squeeze()))
                        past_cos_sim -= self.distill_threshold
                        beta_masking = torch.clamp(past_cos_sim, min=0.0, max=1.0)
                        print("y")
                        print(y)
                        print("beta_masking")
                        print(beta_masking)
                        print("loss", loss.mean(), "l2_loss", (self.distill_beta * beta_masking * l2_loss).mean())
                        loss = (loss + self.distill_beta * beta_masking * l2_loss).mean()
                    
                    elif self.distill_strategy == "classwise_difference":
                        current_feature = torch.stack([self.pre_logits(torch.mean(torch.stack(self.current_cls_feature_dict[y_i.item()], dim=0), dim=0).unsqueeze(0)) for y_i in y], dim=0)
                        past_cos_sim = self.get_cos_sim(target_fc, past_feature.detach().squeeze())
                        current_cos_sim = self.get_cos_sim(target_fc, current_feature.squeeze().to(self.device))
                        masking = torch.abs(past_cos_sim) > torch.abs(current_cos_sim)
                        beta_masking = torch.abs(past_cos_sim)
                        beta_masking -= self.distill_threshold
                        beta_masking = torch.clamp(beta_masking, min=0.0, max=1.0)
                        beta_masking = masking * beta_masking
                        
                        #sim_difference = torch.abs(past_cos_sim - current_cos_sim)
                        #sim_difference -= self.distill_threshold
                        #beta_masking = torch.clamp(sim_difference, min=0.0, max=1.0)
                        print("y")
                        print(y)
                        print("beta_masking")
                        print(beta_masking)
                        print("loss", loss.mean(), "l2_loss", (self.distill_beta * beta_masking * l2_loss).mean())
                        loss = (loss + self.distill_beta * beta_masking * l2_loss).mean()
                
                # residual dict update
                for idx, t in enumerate(y):
                    if t.item() not in self.residual_dict.keys():
                        self.residual_dict[t.item()] = [residual[idx]]
                        self.feature_dict[t.item()] = [feature.detach().cpu()[idx]]
                    else:
                        self.residual_dict[t.item()].append(residual[idx])
                        self.feature_dict[t.item()].append(feature.detach().cpu()[idx])
                
                    if len(self.residual_dict[t.item()]) > self.residual_num:
                        self.residual_dict[t.item()] = self.residual_dict[t.item()][1:]
                        self.feature_dict[t.item()] = self.feature_dict[t.item()][1:]

        # accuracy calculation
        if self.loss_criterion == "DR":
            with torch.no_grad():
                cls_score = feature @ self.etf_vec
                acc, correct = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                #acc, _ = self.compute_accuracy(cls_score[:, self.etf_index_list], y)
                acc = acc.item()
        elif self.loss_criterion == "CE":
            _, preds = logit.topk(self.topk, 1, True, True)
            correct = torch.sum(preds == y.unsqueeze(1)).item()

        return logit, loss, feature, correct

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)

            self.before_model_update()

            self.optimizer.zero_grad()

            # logit can not be used anymore
            _, loss, feature, correct_batch = self.model_forward(x,y)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.after_model_update()

            total_loss += loss.item()
            correct += correct_batch
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data
        
    def etf_initialize(self):
        logger.info("ETF head : evaluating {} out of {} classes.".format(self.eval_classes, self.num_classes))

        orth_vec = self.generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        self.etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1))).to(self.device)

    def get_angle(self, a, b):
        inner_product = (a * b).sum(dim=0)
        a_norm = a.pow(2).sum(dim=0).pow(0.5)
        b_norm = b.pow(2).sum(dim=0).pow(0.5)
        cos = inner_product / (2 * a_norm * b_norm)
        angle = torch.acos(cos)
        return angle

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        rand_mat = np.random.random(size=(feat_in, num_classes))
        orth_vec, _ = np.linalg.qr(rand_mat) # qr 분해를 통해서 orthogonal한 basis를 get
        orth_vec = torch.tensor(orth_vec).float()
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
            "The max irregular value is : {}".format(
                torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
        return orth_vec

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def add_new_class(self, class_name, sample):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample)
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                self.sample_inference(self.temp_batch)
                train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
                self.report_training(sample_num, train_loss, train_acc)
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []
        
        # save feature and etf-fc
        if self.store_pickle and self.rnd_seed == 1:
            if self.sample_num % 100 == 0 and self.sample_num !=0:

                name_prefix = self.note + "/etf_resmem_sigma" + str(self.sigma) + "_num_" + str(self.sample_num) + "_iter" + str(self.online_iter) + "_sigma" + str(self.softmax_temperature) + "_criterion_" + self.select_criterion + "_top_k" + str(self.knn_top_k) + "_knn_sigma"+ str(self.knn_sigma)
                fc_pickle_name = name_prefix + "_fc.pickle"
                feature_pickle_name = name_prefix + "_feature.pickle"
                class_pickle_name = name_prefix + "_class.pickle"
                pickle_name_feature_std_mean_list = name_prefix + "_feature_std.pickle"
                pickle_name_stds_list = name_prefix + "_stds.pickle"

                self.save_features(feature_pickle_name, class_pickle_name)

                with open(fc_pickle_name, 'wb') as f:
                    pickle.dump(self.etf_vec[:, :len(self.memory.cls_list)].T, f, pickle.HIGHEST_PROTOCOL)

                with open(pickle_name_feature_std_mean_list, 'wb') as f:
                    pickle.dump(self.feature_std_mean_list, f, pickle.HIGHEST_PROTOCOL)
                
                with open(pickle_name_stds_list, 'wb') as f:
                    pickle.dump(self.stds_list, f, pickle.HIGHEST_PROTOCOL)

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)


    def sub_simple_test(self, x, softmax=False, post_process=False):
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        #cls_score = cls_score[:, :self.eval_classes]
        cls_score = cls_score[:, :len(self.memory.cls_list)]
        assert not softmax
        '''
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score
        '''
        return cls_score


    def simple_test(self, img, gt_label, return_feature=True):
        """Test without augmentation."""
        '''
        if return_backbone:
            x = self.extract_feat(img, stage='backbone')
            return x
        x = self.extract_feat(img)
        '''
        _, feature = self.model(img, get_feature=True)
        res = self.sub_simple_test(feature, post_process=False)
        res = res.argmax(dim=-1)
        if return_feature:
            return torch.eq(res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist(), feature
        else:
            return torch.eq(res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist()


    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        total_acc = 0.0
        label = []
        feature_dict = {}
        self.model.eval()


        residual_list = []
        feature_list = []
        for key in self.residual_dict.keys():
            residual_list.extend(self.residual_dict[key])
            feature_list.extend(self.feature_dict[key])
        residual_list = torch.stack(residual_list)
        feature_list = torch.stack(feature_list)

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                if self.loss_criterion == "DR":
                    res, features = self.simple_test(x, y, return_feature=True)

                elif self.loss_criterion == "CE":
                    logit, features = self.model(x, get_feature=True)

                features = self.pre_logits(features)

                # |z-z(i)|**2
                w_i_lists = [-torch.norm(feature - feature_list, p=2, dim=1, keepdim=True) for feature in features.detach().cpu()]

                # top_k w_i index select
                w_i_indexs = [torch.topk(w_i_list.squeeze(), self.knn_top_k)[1].long() for w_i_list in w_i_lists]

                # top_k w_i 
                if self.select_criterion == "softmax":
                    w_i_lists = [self.softmax(torch.topk(w_i_list.squeeze(), self.knn_top_k)[0] / self.knn_sigma) for w_i_list in w_i_lists]
                elif self.select_criterion == "linear": # just weight sum
                    w_i_lists = [torch.topk(w_i_list.squeeze(), self.knn_top_k)[0] / torch.sum(torch.topk(w_i_list.squeeze(), self.knn_top_k)[0]).item() for w_i_list in w_i_lists]

                # select top_k residuals
                residual_lists = [residual_list[w_i_index] for w_i_index in w_i_indexs]
                residual_terms = [rs_list.T @ w_i_list  for rs_list, w_i_list in zip(residual_lists, w_i_lists)]

                unique_y = torch.unique(y).tolist()
                for u_y in unique_y:
                    indices = (y == u_y).nonzero(as_tuple=True)[0]
                    if u_y not in feature_dict.keys():
                        feature_dict[u_y] = [torch.index_select(features, 0, indices)]
                    else:
                        feature_dict[u_y].append(torch.index_select(features, 0, indices))


                if self.loss_criterion == "DR":
                    # Add residual terms to ETF head
                    features += torch.stack(residual_terms).to(self.device)
                    target = self.etf_vec[:, y].t()
                    loss = self.criterion(features, target)

                elif self.loss_criterion == "CE":
                    logit = self.softmax(logit / self.softmax_temperature) + torch.stack(residual_terms).to(self.device)
                    loss = criterion(logit, y)

                if self.use_feature_distillation:
                    loss = loss.mean()

                # accuracy calculation
                with torch.no_grad():
                    if self.loss_criterion == "DR":
                        cls_score = features @ self.etf_vec
                        _, correct_count = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                        total_correct += correct_count

                    elif self.loss_criterion == "CE":
                        _, preds = logit.topk(self.topk, 1, True, True)
                        total_correct += torch.sum(preds == y.unsqueeze(1)).item()

                    total_loss += loss.item()
                    total_num_data += y.size(0)

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        #cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc}

        return ret
