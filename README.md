Geometric Transfer Metric Learning (GTML)
=========================================
This package contains the code for the paper ''Ship Classiﬁcation in SAR Images with Geometric Transfer Metric Learning''.

This paper has been accepted by the journal of IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING.

Part 1 - Framework of GTML for zero labeled sample (ZLS) task:
---------------------------------------
![image](image/ZLS_framework.pdf)

The model can be run by executing ZLS_GTML_A_main.m and ZLS_GTML_R_main.m in the code directory.

Part 2 - Framework of GTML for scarce labeled samples (SLS) task:
------------------------------------------
![image](image/SLS_framework.pdf)

The model can be run by executing SLS_GTML_A_main.m and SLS_GTML_R_main.m in the code directory.

Note
=============
All the algorithms are implemented and run using MATLAB 2018b. 

All the experiments are run on an Intel(R) Core(TM) i7-6800K Processor 3.4 GHz machine, with 64 GB RAM and Windows 10 Professional standard operation system. 

If the above conditions cannot be met, please search for hyper-parameters again according to the following scope: [0.001, 0.01, 0.1, 1, 10, 100, 1000].

References
==========
[1] M. Grant and S. Boyd, “CVX: Matlab software for disciplined convex programming, version 2.1,” http://cvxr.com/cvx, Mar. 2014.

Contact
========
If you have any problem about this package, please create an Issue or send us an Email at:

* sky.yongjie.xu@hotmail.com
* haitaolang@hotmail.com

