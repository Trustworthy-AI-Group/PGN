"""Implementation of evaluate attack result."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader
import pretrainedmodels

batch_size = 10

input_csv = './dataset/images.csv'
input_dir = './dataset/images'
adv_dir = './incv3_pgn_outputs'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'inception_v3':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inception_v4':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_50':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_101':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_152':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inc_res_v2':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'tf2torch_adv_inception_v3':
        from torch_nets import tf_adv_inception_v3
        net = tf_adv_inception_v3
        model = nn.Sequential( 
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        from torch_nets import tf_ens3_adv_inc_v3
        net = tf_ens3_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        from torch_nets import tf_ens4_adv_inc_v3
        net = tf_ens4_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        from torch_nets import tf_ens_adv_inc_res_v2
        net = tf_ens_adv_inc_res_v2
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    else:
        print('Wrong model name!')

    return model

def verify(model_name, path):
    img_size = 299
    model = get_model(model_name, path)

    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor(), T.Resize(img_size)]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(1) != (gt)).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))

def verify_ensmodels(model_name, path):
    img_size = 299
    model = get_model(model_name, path)

    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor(), T.Resize(img_size)]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            # print(sum)
            sum += (model(images)[0].argmax(1) != (gt+1)).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))

def main():
    model_names = ['inception_v3', 'inception_v4']
    model_names_ens = ['tf2torch_ens3_adv_inc_v3', 'tf2torch_ens4_adv_inc_v3'] # You can download the pretrained ens_models from https://github.com/ylhz/tf_to_pytorch_model
    models_path = './models/'
    for model_name in model_names:
        verify(model_name, models_path)
        print("===================================================")
    for model_name in model_names_ens: # When we validate the ens model, we should change gt to gt+1 as the ground truth label.
        verify_ensmodels(model_name, models_path)
        print("===================================================")
if __name__ == '__main__':
    print(adv_dir)
    main()

