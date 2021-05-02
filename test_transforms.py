import PIL.Image as Image

import transforms

def resize():
    img = Image.open("./imgs/a.jpg")
    transformer = transforms.Resize(512)
    img1, img2 = transformer(img, img)
    print(img1.size, img2.size)

def random_crop():

    img = Image.open("./imgs/a.jpg")
    
    transformer=transforms.RandomCrop(10)
    img1, img2 = transformer(img, img)
    print(img1.size, img2.size)

def compose():
    transformer = transforms.Compose([
        transforms.Resize([556, 556]),
        transforms.Resize([512, 512]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    img = Image.open("./imgs/a.jpg")
    img1, img2 = transformer(img, img)
    print(img1.shape, img2.shape)


compose()