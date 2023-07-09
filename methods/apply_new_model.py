from methods.er_new import ER
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler


class App(ER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        super().__init__(train_datalist, test_datalist, device, **kwargs)
    
    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.model = select_model(self.model_name, self.dataset, 1, pre_trained=True).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        
        #self.memory = MemoryBase(self.memory_size)
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.temp_future_batch_idx = []
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

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        for i in range(self.future_steps):
            self.load_batch()
    