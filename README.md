# MMNet-PyTorch (ACM MM, 2020)
MMNet: Multi-Stage and Multi-Scale Fusion Network for RGB-D Salient Object Detection.
PDF: [PDF](https://dl.acm.org/doi/10.1145/3394171.3413523)  


## Requirements
•	pytorch 1.3.0+   
•	torchvision   
•	PIL   
•	Numpy   


## Testing
•	Download the trained model from [here](https://pan.baidu.com/s/1sGj6HacGepzWX9-8q8NThQ) [code: ofcn]   
•	Download test datasets from [here](https://pan.baidu.com/s/1hOWEFfcIXGwYHDCzASgkhg ) [code: sva4]  
•	Modify your `test_dataroot` and `test_datasets` in test.py   
•	Test the MMNet: `python test.py`   


## Training 
•	Download the train-augment dataset from [here](https://pan.baidu.com/share/init?surl=8nVAiOkTKczB_ZpIzBHA0A) [code: haxl]    
•	Download the pretrained backbone Res2Net(baseWidth = 48, scale = 2) from [here](https://github.com/Res2Net/Res2Net-PretrainedModels)    
•	Modify your `train_dataroot` and `pre_trained_root` in train.py  
•	Train the MMNet: `python train.py`      


## Results
•	Saliency maps mentioned in the paper can be download from [here](https://pan.baidu.com/s/1S2ZT1AGqW0CfwaGFmubbbQ) [code: wl4s]  
[1] The test_Results are obtained by trained on NJUD & NLPR & DUT (1485+700+800).     
[2] The test_results_COME_train are obtained by trained on the new COME-Train dataset (8025).    
•	The saliency results can be evaluated by using the tool in [Matlab](http://dpfan.net/d3netbenchmark/)   


## Citation
Please cite our paper if you use this repository in your reseach.
```
@inproceedings{MMNet20,   
author = {Liao, Guibiao and Gao, Wei and Jiang, Qiuping and Wang, Ronggang and Li, Ge},  
title = {MMNet: Multi-Stage and Multi-Scale Fusion Network for RGB-D Salient Object Detection},  
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},   
pages = {2436–2444},   
year = {2020}
}  
```
