#!/bin/bash

device=0
TCLayer=ST2LiteMBConv
tcr=4

#configCode=9552
#dataset="3mdad-rgb1"
#echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
#python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
#             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3
#
configCode=2561
dataset="3mdad-rgb2"
echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3

#configCode=144
#dataset="ebdd"
#echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
#python3 ../main_softnode2_fixed.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
#             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  -gcmh 3

