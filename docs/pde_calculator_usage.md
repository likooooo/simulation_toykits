1. 边界条件与网格定义
- 周期性边界条件 : 网格点上
- 其他(D->D, N->N, D->N, N->D) : 网格中心 参考 : test_sturm_liouville.py

2. 计算稳定性
- input 数据的边界条件要与坐标轴定义一致
- 步长需要结合输入数据的截止频率进行设置， 参考 : test_sturm_liouville_nyquist.py

3. 非齐次边界条件
使用齐次化原理将边界条件齐次化之后再求解

[正确性验证]

介绍 SL 理论
https://en.wikipedia.org/wiki/Sturm%E2%80%93Liouville_theory

目前支持任意维度的 p, q是grid indepency 的常量
如果 p, q是与位置相关的函数, 那么计算量会从N-> N^2
后续会加入1d/2d的 p, q是与位置相关的函数的支持。
当 p 是与位置相关的函数, L 可以包含一阶项


介绍如何上传文件


example:
波动方程

example :
bessel/legendre equation

