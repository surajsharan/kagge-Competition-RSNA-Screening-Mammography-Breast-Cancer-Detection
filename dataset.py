import config
import cv2
import torch 
from torch.utils.data.distributed import DistributedSampler
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from ignite.distributed import DistributedProxySampler
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import numpy as np
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class RSNADataSet(torch.utils.data.Dataset):
    def __init__(self, df, image_path, transforms=None, test=False):
        super().__init__()
        self.df = df
        self.path = image_path
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        
        image = cv2.imread(self.df.iloc[item][self.path])
        
        
        if config.LOCAL_BINARY_PATTERN:
        
            gray = rgb2gray(image.copy())
            nw_img = local_binary_pattern(gray.copy(), P=8, R=5, method='ror')
            nw_img_bin = local_binary_pattern(gray.copy(), P=8, R=5, method='default')
            # image = np.stack((gray, nw_img, nw_img_bin), axis=-1)
            image = np.dstack([gray, nw_img, nw_img_bin])

        
        if self.transforms is not None:
            if self.df.iloc[item][config.TARGET] > 0:
                image = config.TARGET_TRANSFORM(image=image)['image']
            else:
                image = self.transforms(image=image)['image']

            
        
            
        if self.test:
            return {
                "image": image,
                "meta":torch.as_tensor(self.df.iloc[item][config.CATEGORY_AUX_TARGETS])
                    }
        else:
            return {
                "image":image,
                "meta":torch.as_tensor(self.df.iloc[item][config.CATEGORY_AUX_TARGETS]),
                "target":torch.as_tensor(self.df.iloc[item][config.TARGET]),
                "image_id": torch.as_tensor(self.df.iloc[item]['image_id']),
                    }

class RSNADataloader:
    def __init__(self, df,image_path, weights, transforms=None, test=False):
        self.df = df
        self.path = image_path
        self.transforms = transforms
        self.test = test
        self.weights = weights
        self.dataset = RSNADataSet(
            df=self.df, image_path=self.path,transforms=self.transforms, test=self.test)

    def fetch(self, batch_size, num_workers, drop_last=False, shuffle=True,ddp=None,validation=False):
        if not ddp:
            if validation:
                sampler = None
                data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,pin_memory=True)
            else:
                sampler = ExhaustiveWeightedRandomSampler(self.weights, len(self.weights))
                data_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,pin_memory=True)

        else:
            rank , world_size = ddp
            if validation:
                sampler = None
                distributed_sampler = DistributedSampler(self.dataset, rank=rank, num_replicas=world_size,shuffle=shuffle)
            
            else:
                sampler = ExhaustiveWeightedRandomSampler(self.weights, len(self.weights))
                distributed_sampler = DistributedProxySampler(sampler,rank=rank, num_replicas=world_size)
            
            # distributed_sampler = DistributedSampler(self.dataset, rank=rank, num_replicas=world_size,shuffle=shuffle,sampler=sampler)
            data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=distributed_sampler, num_workers=num_workers, drop_last=drop_last)
                        # worker_init_fn=lambda worker_id: distributed_sampler.set_epoch(epoch))
            
        return data_loader
        

        
class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch