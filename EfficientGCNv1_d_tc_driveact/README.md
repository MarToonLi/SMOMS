2023年4月14日
- 目测，当前路径下的sh文件，近test_val0.sh,test_val1.sh,test_val2.sh,和test_test0.sh,test_test1.sh,test_test2.sh 是有价值的。证据如下：
  split1:
  ```
    #python3 ../main_softnode3_test.py  --datasetCode driveract-0   --configCode  1150   --device  0  --granularity mid
    #python3 ../main_softnode3_test.py  --datasetCode driveract-0   --configCode  3082   --device  0  --granularity task
    #python3 ../main_softnode3_test.py  --datasetCode driveract-0   --configCode  5373   --device  0  --granularity action
    #python3 ../main_softnode3_test.py  --datasetCode driveract-0   --configCode  8642   --device  0  --granularity object
    #python3 ../main_softnode3_test.py  --datasetCode driveract-0   --configCode  7470   --device  0  --granularity location
    #python3 ../main_softnode3_test.py  --datasetCode driveract-0   --configCode  7337   --device  0  --granularity all
    
    #python3 ../main_softnode3.py  --datasetCode driveract-0   --configCode  1150   --device  0  --granularity mid
    #python3 ../main_softnode3.py  --datasetCode driveract-0   --configCode  3082   --device  0  --granularity task
    #python3 ../main_softnode3.py  --datasetCode driveract-0   --configCode  5373   --device  0  --granularity action
    #python3 ../main_softnode3.py  --datasetCode driveract-0   --configCode  8642   --device  0  --granularity object
    #python3 ../main_softnode3.py  --datasetCode driveract-0   --configCode  7470   --device  0  --granularity location
    #python3 ../main_softnode3.py  --datasetCode driveract-0   --configCode  7337   --device  0  --granularity all
  ```
  split2:
  ```
    #python3 ../main_softnode3_test.py  --datasetCode driveract-1   --configCode  1150   --device  1  --granularity mid
    #python3 ../main_softnode3_test.py  --datasetCode driveract-1   --configCode  7456   --device  1  --granularity task
    #python3 ../main_softnode3_test.py  --datasetCode driveract-1   --configCode  7470   --device  1  --granularity action
    #python3 ../main_softnode3_test.py  --datasetCode driveract-1   --configCode  2561   --device  1  --granularity object
    #python3 ../main_softnode3_test.py  --datasetCode driveract-1   --configCode  144   --device  2  --granularity location
    #python3 ../main_softnode3_test.py  --datasetCode driveract-1   --configCode  5374   --device  1  --granularity all
    #python3 ../main_softnode3_test.py  --datasetCode driveract-1   --configCode  1478   --device  0  --granularity object
  
    #python3 ../main_softnode3.py  --datasetCode driveract-1   --configCode  1150   --device  1  --granularity mid
    #python3 ../main_softnode3.py  --datasetCode driveract-1   --configCode  7456   --device  1  --granularity task
    #python3 ../main_softnode3.py  --datasetCode driveract-1   --configCode  7470   --device  1  --granularity action
    #python3 ../main_softnode3.py  --datasetCode driveract-1   --configCode  2561   --device  1  --granularity object
    #python3 ../main_softnode3.py  --datasetCode driveract-1   --configCode  144   --device  2  --granularity location
    #python3 ../main_softnode3.py  --datasetCode driveract-1   --configCode  5374   --device  1  --granularity all
    #python3 ../main_softnode3.py  --datasetCode driveract-1   --configCode  1478   --device  0  --granularity object
  ```
  split3:
  ```
    #python3 ../main_softnode3_test.py  --datasetCode driveract-2   --configCode  2561   --device  2  --granularity mid
    #python3 ../main_softnode3_test.py  --datasetCode driveract-2   --configCode  9954   --device  2  --granularity task
    #python3 ../main_softnode3_test.py  --datasetCode driveract-2   --configCode  7456   --device  2  --granularity action
    #python3 ../main_softnode3_test.py  --datasetCode driveract-2   --configCode  144   --device  2  --granularity object
    #python3 ../main_softnode3_test.py  --datasetCode driveract-2   --configCode  4059   --device  2  --granularity location
    #python3 ../main_softnode3_test.py  --datasetCode driveract-2   --configCode  4059   --device  2  --granularity all
    #python3 ../main_softnode3_test.py  --datasetCode driveract-2   --configCode  4059   --device  2  --granularity object
  ```

2023年4月16日  
- test_special系列是面向main_softnode3_special.py
- test_test系列则是面向main_softnode3_test.py和main_softnode3.py
- test_val系列则是面向main_softnode3_test_step.py、main_softnode3_test.py和main_softnode3.py
- main_softnode3.py 和 main_softnode3_GPUs.py(已删除)对比可知，后者缺乏granularity参数，因为被视为落后的版本，因此舍去。
- main_softnode3.py 和 main_softnode3_test.py 代码对比可知，两者的代码是一致的。test版本可能是希望作为尝试用。
- main_softnode3_test.py 和 main_softnode3_test_step.py 代码对比可知,两者的代码唯一的重要区别在于各个随机种子下的参数从余弦变化转化为step变化。
- main_softnode3_test_step.py 和 main_softnode3_special.py 代码对比可知,两者的代码也基本一致。不知道为啥、最大原因感觉是希望做备份。
- 查看scripts文件夹下的sh文件，尤其是test.sh和val.sh结尾的六个文件，
  **认为main_softnode3.py是主要的**，main_softnode3_test.py，main_softnode3_test_step.py，main_softnode3_special.py是为了测试用的。
- **Drive&Act版和3MDAD及EBDD版的差别主要是模型以外的部分，尤其是数据集的获取，产生了很多变量以及用于跨函数传递的变量/字典**。
- 在Drive&Act18个任务的随机种子的确定时，发现object的split0和1的val值和test值没有存在于 一审_调参数据2_迁徙_第一篇论文的模型试验结果.xlsx文件中。
  但是出现在0527的文件(evaluate_splits_0527.py)中，意味着那四个值是在step方法下计算得到的。
  ```python
    split0_path = os.path.join(basePath, "valid0526_object_driveract-0_144_data.npy")
    split1_path = os.path.join(basePath, "valid0526_object_driveract-1_144_data.npy")
    split2_path = os.path.join(basePath, "valid0526_object_driveract-2_4059_data.npy")
    # 70.06399904059727 56.92715984499602 55.95802529882721 57.883782453003185
    # 70.06399904059727 56.92715984499602 55.62781087294128 57.84198980144436
  ```
- resource0527和evaluate_splits_0527以及main_softnode3_test_step.py的解释和意义： 
  之所以进行第二阶段的实验(0527)，是因为基于余弦的结果中，object的val值和location的val值结果不好。 
  而最后探究的结果，中obejct的split0和split1的随机种子得到改善。同时location的split2的随机种子也得到了改善。
  
- 目前除了复现不出来，基本没有大问题了。