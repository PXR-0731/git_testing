import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model

from clearml import Dataset

# def load_model(model_weight_path):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # # create model
#     # model = create_model(num_classes=6, has_logits=False).to(device)
#     # # load model weights
#     # cur_dir = ('/').join(os.path.abspath(__file__).split('\\')[:-1])
#     # model_weight_path = os.path.join(cur_dir, "weights/model-29.pth")
#     model.load_state_dict(torch.load(model_weight_path, map_location=device))
#     model.eval()
#     return model, device


def predict(model, device, img_path:str):

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    # img_path = "./predicting_photo.jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r", encoding='utf-8') as f:
        class_indict = json.load(f)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # print_res = {'class': class_indict[str(predict_cla)], 'prob': predict[predict_cla].numpy()}
    # print_res = json.dumps(print_res)

    # plt.title(print_res, fontproperties="SimHei")
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # plt.show()
    return print_res


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    # cur_dir = ('/').join(os.path.abspath(__file__).split('\\')[:-1])
    # img_path = os.path.join(cur_dir, "20.png")
    img_path = "./vit/20.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = "./vit/class_indices.json"# os.path.join(cur_dir, 'class_indices.json')
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r", encoding='utf-8') as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=6, has_logits=False).to(device)

    weights_dir = Dataset.get(dataset_id=args.dataset).get_local_copy()
    # load model weights
    model_weight_path = os.path.join(weights_dir, 'model-29.pth') # os.path.join(cur_dir, "weights/model-29.pth")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res, fontproperties="SimHei")
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))
    plt.show()


if __name__ == '__main__':

    from argparse import ArgumentParser
    # adding command line interface, so it is easy to use
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='cb3e81df0c904d238bb73ec213dac9df', type=str, help='Dataset ID to train on')
    args = parser.parse_args()
    main(args)