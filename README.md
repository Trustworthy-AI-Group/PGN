# PGN
This repository contains the PyTorch code for the paper:

**Boosting Adversarial Transferability by Achieving Flat Local Maxima [NeurIPS 2023]**.

Zhijin Ge, Fanhua Shang, Hongying Liu, Yuanyuan Liu, Xiaosen Wang.

![loss surface map](https://github.com/Trustworthy-AI-Group/PGN/blob/main/doc/images/loss_surface.png)
## Requirements
* Python == 3.7.11
* pytorch == 1.8.0
* torchvision == 0.8.0
* numpy == 1.21.2
* pandas == 1.3.5
* opencv-python == 4.5.4.60
* scipy == 1.7.3
* pillow == 8.4.0
* pretrainedmodels == 0.7.4
* tqdm == 4.62.3
* imageio == 2.6.1


## Qucik Start
### Prepare the dataset and models.
1. You can download the ImageNet-compatible dataset from [here](https://github.com/Zhijin-Ge/STM/tree/main/dataset) and put the data in **'./dataset/'**.

2. The normally trained models (i.e., Inc-v3, Inc-v4, IncRes-v2, Res-50, Res-101, Res-100) are from "pretrainedmodels", if you use it for the first time, it will download the weight of the model automatically, just wait for it to finish. 

3. The adversarially trained models (i.e, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2) are from [SSA](https://github.com/yuyang-long/SSA) or [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model). For more detailed information on how to use them, visit these two repositories.

### Runing attack
1. You can run our proposed attack as follows. 
```
python Incv3_PGN_Attack.py
```
2. The generated adversarial examples would be stored in the directory **./incv3_xx_xx_outputs**. Then run the file **verify.py** to evaluate the attack success rate of each model used in the paper:
```
python verify.py
```
3. You can run the file **'surface_map.py'** to visualize the loss surface maps for the adversarial examples, the maps will be stored in the directory **'./loss_surfaces/'**.
```
python surface_map.py
```
## Citation
If our paper or this code is useful for your research, please cite our paper.
```
@inproceedings{ge2023boosting,
     title={{Boosting Adversarial Transferability by Achieving Flat Local Maxima}},
     author={Zhijin Ge and Fanhua Shang and Hongying Liu and Yuanyuan Liu and Xiaosen Wang},
     booktitle={Proceedings of the Advances in Neural Information Processing Systems},
     year={2023},
}
```
