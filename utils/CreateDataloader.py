import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset


def get_class_list(pathe):
    class_list = []
    with open(pathe, "r", encoding="utf-8") as f:
        for line in f:
            if line[-3] == '\t':
                class_list.append(int(line[-2]))
            else:
                class_list.append(int(line[-3:-1]))
    return class_list

class LoadData(Dataset):
    def __init__(self, txt_path, tranf_train, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        self.tranf_train = tranf_train
        self.targets = get_class_list(txt_path)
        # self.tranf_train2 = tranf_train2
        # self.tranf_test = tranf_test

        # self.train_tf = transforms.Compose([
        #         transforms.Resize(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.ToTensor(),
        #         transform_BZ
        #     ])
        # self.val_tf = transforms.Compose([
        #         transforms.Resize(224),
        #         transforms.ToTensor(),
        #         transform_BZ
        #     ])
        # self.train_tf = transforms.Compose([
        #     transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(),
        #     transform_BZ,
        # ])
        # self.val_tf = transforms.Compose([
        #     transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(),
        #     transform_BZ,
        # ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info

    def padding_black(self, img):
        w, h  = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        # img = self.padding_black(img)
        if self.train_flag:
            img = self.tranf_train(img)
        else:
            img = self.tranf_test(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs_info)


if __name__ == "__main__":
    train_dataset = LoadData("train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(image)
        # img = transform_BZ(image)
        # print(img)
        print(label)
