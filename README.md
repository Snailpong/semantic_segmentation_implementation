# semantic_segmentation_implementation
PyTorch implementation of semantic segmentation models

## Models
- [FCN](https://arxiv.org/pdf/1411.4038.pdf) (FCN-8s)
- [DeconvNet](https://arxiv.org/pdf/1505.04366.pdf)
- [PSPNet](https://arxiv.org/pdf/1612.01105.pdf) (ResNet18 bottleneck)

## Dependencies
- Pytorch
- torchvision
- numpy
- PIL
- tqdm
- click

## Dataset Settings
I used [VOC2012 dataset](http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar) with [additonal labeled data](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0).
```
.
└── data
    └── VOCdevkit
         └── VOC2012
            ├── JPEGImages
            |    ├── 2007_000027.jpg
            |    └── ...
            └── SegmentationClassAug
                 ├── 2007_000027.png
                 └── ...
```

## Train & Test
```
python train.py --model_name=[fcn, deconvnet, pspnet]
python test.py --model_name=[fcn, deconvnet, pspnet] --image_path=(PATH)
```

## Result
- PSPNet evaluation with PSPNet trained 36 epochs (loss: 0.47 ~ 0.50)
<table style="text-align: center">
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/111900901-6cbd8380-8a78-11eb-8c5d-5be76d7004b7.jpg" alt="1"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/111900902-6d561a00-8a78-11eb-934e-adce4a424ed5.jpg" alt="1"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/111900903-6deeb080-8a78-11eb-8607-eac1bfc59918.jpg" alt="1"></td>
</tr>
<tr><td>Input image</td><td>True label</td><td>Estimated</td></tr>
</table>

## Reference
- https://github.com/HyeonwooNoh/DeconvNet
- https://github.com/Lextal/pspnet-pytorch