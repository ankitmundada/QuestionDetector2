from torchvision import transforms
from base import BaseDataLoader
from dataset import datasets
from dataset.transforms import ToTensor


class FER2013Loader(BaseDataLoader):
    """
    FER2013 Loader
    """
    def __init__(self, data_dir, batch_size, shuffle, num_workers):

        self.dataset = datasets.FER2013Dataset(data_dir, transform=transforms.Compose([ToTensor()]))

        super(FER2013Loader, self).__init__(self.dataset, batch_size, shuffle, num_workers)
