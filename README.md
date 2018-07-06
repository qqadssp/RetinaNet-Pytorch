# RetinaNet

# Train on VOC  

1.Download PASCAL VOC 2012 trainval datasets and unzip it. Its path should be '{root_dir}/VOCdevkit/..'  

2.Download this repo  

    git clone git@github.com:qqadssp/RetinaNet.git  

    cd RetinaNet  

3.Download pretrained weights from https://download.pytorch.org/models/resnet50-19c8e357.pth  

    cd checkpoint  

    wget https://download.pythorch.org/models/resnet50-19c8e357.pth  
 
    cd ..  

4.Initialize the model  

    python init.py  

5.Modify configs file in 'config'. For VOC datatsets, modify 'TRAIN: DATASETS_DIR' with your {root_dir}  

6.Trian the model 

    python train.py --cfg ./configs/RetinaNet_ResNet50_FON_VOC.yaml  

# Test on VOC  

1.Download PASCAL VOC 2012 test datasets and unzip it. Its path should be '{root_dir}/VOCdevkit_test/..'  

2.Modify ocnfig file in 'configs'. For VOC datasets, modify 'TEST: DATASETS_DIR' with your {root_dir}, and 'TEST: WEIGHTS' with the trained weights in 'checkpoint'  

3.Test the model. The result files will be in 'result'.  

    python test.py --cfg ./configs/RetinaNet_ResNet50_FPN_VOC.yaml  
