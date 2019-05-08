import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, item):
        image, label = item['image'], item['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #print(image.shape)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.tensor(label)}
