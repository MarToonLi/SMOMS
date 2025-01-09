#!/bin/bash

device=0
TCLayer=ST2LiteMBConv
tcr=4


configCode=2561
dataset="3mdad-rgb2"

lingshen_modelType="EfficientGCN-B0"
yishen2_modelType="EfficientGCN-Y2-B0"
yishen3_modelType="EfficientGCN-Y3-B0"
yishen4_modelType="EfficientGCN-Y4-B0"
yishen5_modelType="EfficientGCN-Y5-B0"

echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${lingshen_modelType}

echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen2_modelType}

echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen3_modelType}

echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen4_modelType}

echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3 -mt ${yishen5_modelType}