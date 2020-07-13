# ALEN
Attention-based network for low-light image enhancement

---

This is a pytorch implementation of ALEN in ICME2020

[Paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102774)

This is the link of pre-trained model and test results in [OneDrive](https://1drv.ms/u/s!ArN5fsqvewP_gn-9aXHBTnIn9UuM?e=UofBNZ) and [BaiduNetDisk](https://pan.baidu.com/s/1nCxNIllzcF0FxCKf4MoXvg)(2fex).

---

## Test
1. download the pre-trained model
2. prepare test dataset
3. run : python main.py --phase=test

requirement: 
python == 3.7 
pytorch == 1.0.0
rawpy

You need change the dataset path when testing your images, and you need pay attention to the pack function in dataset.py when you test raw images captured by other devices.

---

## Train

prapare the dataset and train the network. That's all.

---

## Citation
If you use our code, please cite our paper




