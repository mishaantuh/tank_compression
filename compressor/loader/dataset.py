import os
import matplotlib.image as img
import torchvision.transforms as transforms

from torch.utils.data import Dataset


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class DatasetLoader(Dataset):
    def __init__(self, path, image_names):
        super().__init__()
        self.image_names = image_names
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image
