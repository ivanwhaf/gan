import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, root, type_='train', transforms=None, class_label_dct=None, num_per_class=None,
                 train_proportion=1.0, valid_proportion=0.1):
        super(ImgDataset, self).__init__()
        self.dataset = []
        self.label = []
        self.transforms = transforms
        self.class_label_dct = class_label_dct
        self.num_per_class = num_per_class

        img_classes = os.listdir(root)  # class name list
        for img_class in img_classes:
            dataset_t, label_t = [], []
            img_class_path = os.path.join(root, img_class)
            print(img_class)

            imgs = os.listdir(img_class_path)
            if self.num_per_class is not None:
                imgs = imgs[:self.num_per_class]

            for img in imgs:
                img_path = os.path.join(img_class_path, img)
                dataset_t.append(img_path)
                label_t.append(class_label_dct[img_class])

            train_per_class = int(len(dataset_t) * train_proportion)
            valid_per_class = int(len(dataset_t) * valid_proportion)
            # test_per_class = int(len(dataset_t) * test_proportion)

            if type_ == 'train':
                dataset_t = dataset_t[:train_per_class]
                label_t = label_t[:train_per_class]
            elif type_ == 'val':
                dataset_t = dataset_t[train_per_class:train_per_class + valid_per_class]
                label_t = label_t[train_per_class:train_per_class + valid_per_class]
            elif type_ == 'test':
                dataset_t = dataset_t[train_per_class + valid_per_class:]
                label_t = label_t[train_per_class + valid_per_class:]

            self.dataset.extend(dataset_t)
            self.label.extend(label_t)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        # img = cv2.imdecode(np.fromfile(
        #     img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = Image.open(img_path).convert('RGB')  # must convert to RGB!
        label = self.label[index]
        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.dataset)
