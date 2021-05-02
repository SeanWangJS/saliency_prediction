import transforms
from dataset import SaliconDataset

images_dir = "./data_sample/images"
fixations_dir = "./data_sample/fixations"

train_csv = "./data_sample/train.csv"
transformer = transforms.Compose([
    transforms.Resize(556),
    transforms.RandomCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset=SaliconDataset(images_dir, fixations_dir, train_csv, transformer)
for i in range(len(dataset)):
    img, fixations = dataset[i]
    print(img.shape, fixations.shape)
    