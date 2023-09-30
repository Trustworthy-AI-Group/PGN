"""Loss surface maps."""
import os
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--adv_dir', type=str, default='./incv3_pgn_outputs/', help='Output directory with adversarial examples.')
parser.add_argument('--output_path', type=str, default='./loss_surfaces/', help='Output directory with loss surfaces.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--batch_size", type=int, default=1, help="How many images process at one time.")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def img2torch(image):
    pil_img = Image.open(image).convert('RGB')
    image = transforms(pil_img)
    image = torch.unsqueeze(image, 0)
    return image

def get_loss_vale(image, model, gt):
    model.eval()
    output = model(image)
    loss_value = F.cross_entropy(output[0].unsqueeze(0), gt)
    return loss_value.detach().item()

def plot_3d_images(img, model, gt, output_path, image_id, adv_dir):
    if os.path.exists(output_path):
        pass
    else:
        os.makedirs(output_path)

    image = os.path.join(adv_dir, image_id)
    images_new = img2torch(image).cuda()
    # two random direction.
    eta = torch.rand_like(img).cuda()
    delta = torch.rand_like(img).cuda()

    a = np.arange(-1, 1, 0.05)
    b = np.arange(-1, 1, 0.05)
    map_3d = np.zeros(shape=(a.shape[0], b.shape[0]))
    size = a.shape[0]

    print(images_new.shape)

    for i in range(size):
        for j in range(size):
            new_image = images_new + 255.0/255*(a[i]*eta+b[j]*delta)
            map_3d[i][j] = get_loss_vale(new_image, model, gt)
    
    X, Y = np.meshgrid(a, b, indexing='ij')
    fig = plt.figure(figsize=(20, 20), facecolor='white')

    sub = fig.add_subplot(111, projection='3d')
    surf = sub.plot_surface(X, Y, map_3d, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False) 
    sub.zaxis.set_major_locator(LinearLocator(10))
    sub.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    sub.set_title(r'PGN', fontsize=40)
    # sub.set_xlabel(r"x axis")
    # sub.set_ylabel(r"y axis")
    # sub.set_zlabel(r"z axis")
    # cb = fig.colorbar(surf, shrink=0.8, aspect=15)
    plt.savefig(os.path.join(output_path, image_id)+'.jpg', dpi=300)


def main():

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda() 
        images = images.cuda()
        plot_3d_images(images, model, gt, output_path=opt.output_path, image_id=images_ID[0], adv_dir=opt.adv_dir)
        
if __name__ == '__main__':
    main()
