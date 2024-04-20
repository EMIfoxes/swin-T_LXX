import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model,seq_SwinTransformer
import hdf5storage

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # load image
    img_path = args.img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    data = hdf5storage.loadmat(img_path)['single_data']
    data = torch.tensor(data,dtype=torch.float32).reshape(125,128).unsqueeze(0)
    data = torch.unsqueeze(data, dim=0)
   
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = seq_SwinTransformer(in_chans=3,
                                patch_size=4,
                                window_size=7,
                                embed_dim=96,
                                depths=(2, 2, 6, 2),
                                num_heads=(3, 6, 12, 24),
                                num_classes=args.num_classes,
                                ).to(device)

    # load model weights
    model_weight_path = args.model_weight_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(data.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # 训练设备类型
    parser.add_argument('--img_path', default='D:/DL_Data/UVA/MF4/MF4_91719.mat', type=str)
    # 训练数据集的根目录
    parser.add_argument('--model_weight_path', default='weights/seq_predict/2024-04-20_21_56/model-last_0.pth', type=str)
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=9, type=int, help='num_classes')
    args = parser.parse_args()
    main(args)
