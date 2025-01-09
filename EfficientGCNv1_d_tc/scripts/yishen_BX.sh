#!/bin/bash

device=0
TCLayer=ST2LiteMBConv
tcr=4

configCode=2561
dataset="3mdad-rgb2"

yishen_B0_modelType="EfficientGCN-B0"
yishen_B1_modelType="EfficientGCN-B1"
yishen_B2_modelType="EfficientGCN-B2"
yishen_B4_modelType="EfficientGCN-B4"
yishen_B3_modelType="EfficientGCN-B3"

echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen_B0_modelType}
echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen_B1_modelType}
echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen_B2_modelType}
echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen_B4_modelType}
echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen_B3_modelType}