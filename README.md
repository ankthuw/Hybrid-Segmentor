## Hybrid-Segmentor: A Hybrid Approach to Automated Damage Detection on Civil Infrastructure

![](./figures/model_architecture.png)

 - **Model**:

You can download Hybrid-Segmentor model weights from [this link](https://1drv.ms/u/s!AtFigR8so_SspuEIg4jDbJNfgGGjyA?e=RNcOGu)

 - **Dataset**:

We created a benchmark dataset called CrackVision12K, which contains cracks. The dataset is developed with 13 publicly available datasets that have been refined using image processing techniques.

**Please note that the use of our dataset is RESTRICTED to non-commercial research and educational purposes.**

You can download the dataset from [this link](https://onedrive.live.com/?authkey=%21AAqG9xQnIlHYoyo&cid=ACF4A32C1F8162D1&id=ACF4A32C1F8162D1%21163379&parId=root&o=OneUp).
|Folder|Sub-Folder|Description|
|:----|:-----|:-----|
|`train`|IMG / GT|RGB images and binary annotation for training|
|`test`|IMG / GT|RGB images and binary annotation for testing|
|`val`|IMG / GT|RGB images and binary annotation for validation|


 - **Reference**:

If you use this dataset for your research, please cite our paper:


```
@article{AEL_dataset,
  title={Automatic crack detection on two-dimensional pavement images: An algorithm based on minimal path selection},
  author={Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={10},
  pages={2718--2729},
  year={2016},
  publisher={IEEE}
}
```

If you have any questions, please contact me: june.goo.21 @ ucl.ac.uk without hesitation.
