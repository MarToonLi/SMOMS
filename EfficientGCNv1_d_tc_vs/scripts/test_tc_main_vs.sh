#!/bin/bash

device=0
TCLayer=ST2LiteMBConv
tcr=4

configCode=2561
dataset="3mdad-rgb2"
echo "====> ${dataset}  ====>  ${TCLayer} ====>  ${tcr}  =="
python3 ../main_softnode2_fixed_vs.py --device ${device}  --configCode ${configCode}   --datasetCode ${dataset} \
             -tcl  Mta2Wrapper-${TCLayer}    -tcr ${tcr}  -ad -ws  --extract   --visualize  -gcmh 3  \
             --visualization_frames  1 3 5 7 9 11 13 15 17  --visualization_sample 3

#    args.generate_data = False
#    args.extract = True
#    args.visualize = True