import os

'''
predict
'''

'''
headle the test_dataset
'''
def handle_test_data(root: str):

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4, ensure_ascii=False)
    # with open('class_indices.json', 'w', encoding='utf-8') as json_file:
    #     json_file.write(json_str)

    # train_images_path = []  # 存储训练集的所有图片路径
    # train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        # val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            # if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            #     val_images_path.append(img_path)
            #     val_images_label.append(image_class)
            # else:  # 否则存入训练集
            #     train_images_path.append(img_path)
            #     train_images_label.append(image_class)
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    # print("{} images for training.".format(len(train_images_path)))
    # print("{} images for validation.".format(len(val_images_path)))
    print("{} images for testing.".format(len(test_images_path)))
    # assert len(train_images_path) > 0, "number of training images must greater than 0."
    # assert len(val_images_path) > 0, "number of validation images must greater than 0."
    assert len(test_images_path) > 0, "number of testing images must greater than 0."
    
    return test_images_path, test_images_label



from torchvision import transforms
from my_dataset_predict import MyDataSet
import torch
from tqdm import tqdm
import sys

def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    wrong_data = []
    for step, data in enumerate(data_loader):
        img_paths, images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        
        compare = torch.eq(pred_classes, labels.to(device))
        for i in range(pred_classes.size(0)):
            if compare[i] == False:
                # message = []
                # message.append(img_paths[i], pred_classes[i], labels[i])
                wrong_data.append([img_paths[i], pred_classes[i].item(), labels[i].item()])
        
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[testing epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, wrong_data


if __name__=="__main__":

    test_images_path, test_images_label = handle_test_data('\\\\192.168.1.77\\share\\projects\\TESTING_DATA')

    data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                             images_class=train_images_label,
    #                             transform=data_transform["train"])

    # 实例化验证数据集
    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["val"])
    batch_size = 8
    nw = min([os.cpu_count(), 8])
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            ) # collate_fn=test_dataset.collate_fn
    
    for i, j in enumerate(val_loader):
        if i==0:
            print(type(j[0]))


    from vit_model import vit_base_patch16_224_in21k as create_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    model = create_model(num_classes=6, has_logits=False).to(device)
    # load model weights
    cur_dir = '/'.join(os.path.abspath(__file__).split('\\')[0:-1])
    model_weight_path = os.path.join(cur_dir, "weights/model-29.pth")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        test_loss, test_acc, wrong_data = evaluate(model=model,
                                        data_loader=val_loader,
                                        device=device,
                                        epoch=1)
    print(test_loss, test_acc)

    sum = 0
    # import shutil
    for message in wrong_data:
        # if message[2]==0:
        #     sum += 1
        # shutil.copy(message[0], 'C:\\Users\\888\\Desktop\\Nothing')
        print(message)
    print(sum)