## Paper
ROSMRF: Real-time Online 3D Semantic segmentation for mobile scene via Multi-modal Representation and Fusion

## System requirements
System : ubuntu 16.04  
Hardware: Quadro RTX 6000  
Software: Pytorch 1.3.0, CUDA-10.0, python 3.5+  For more packages information, we prepared them in the requires.txt file. 
## Useage
1.clone the SSF-DAN repository:  
```
git clone https://github.com/Biotan/ROSMRF
```
2.prepare environment and compile the repository:  
```
virtualenv -p /usr/bin/python3.5 ~/virtualenv/ROSMRF
source ~/virtualenv/ROSMRF/bin/activate
pip install -r requires.txt
bash develop.sh
```
3.download the row data of S3DIS dataset and use the script to preprocess:
```
cd S3DIS/
python preprocess.py
python prepare_data.py
```
4.train the network
```
python unet.py
```
5.evaluate the network
```
python unet_val.py
```
6.visualization the network
```
python unet_vis.py
```
## Pipeline
<img src="https://github.com/Biotan/ROSMRF/blob/main/S3DIS/Img/reconstruct.png" width="800" /><br/>
## Joint 2D-3D framework
<img src="https://github.com/Biotan/ROSMRF/blob/main/S3DIS/Img/framework.png" width="800" /><br/>
## Result
<img src="https://github.com/Biotan/ROSMRF/blob/main/S3DIS/Img/S3DIS.png" width="1000" /><br/>

