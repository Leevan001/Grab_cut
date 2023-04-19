# Grab_cut
Grab的python代码复现


# 用到的库

import numpy as np

import cv2

import maxflow

# 代码略述

一共写了两个版本的迭代

Gbv1是只迭代一轮的代码

Gbv2的迭代是每次都用kmeans重新聚类

Gbv2的迭代是严格按照原论文的迭代，运行速度较慢

Gbv3是打算加快运行速度的，但并没有实现，目前运行速度可提升的方向是constructGCGraph函数

及构建图速度较慢，可以考虑numpy的向量化或多线程，但多线程效果好像并不理想（一次迭代需要6~10s，一半迭代1次效果就比较理想了）

# 代码使用说明：

运行代码首先人为给定框，按下enter或space键确认

![](https://fastly.jsdelivr.net/gh/Leevan001/pictureBed@main/utools/16818782369041681878236688.png)

运行效果图：

![](https://fastly.jsdelivr.net/gh/Leevan001/pictureBed@main/utools/1681877296937feb07b7032b61814a0b9e53e68756ac.png)

# 调试过程

首先是无符号整型数溢出不会报错

calcBeta，在这个函数中遇到了越界的情况，要将uint8转成astype(np.float32)

其次，因为参数gamma 设置地不好，源码中设置为50，我10~50地测试效果都不理想，心态破防，后来设为2，code奇迹般的work了。

换了几个最大流的库，发现pymaxflow最快

聚类直接用opencv的接口最快

能用numpy向量操作的尽量用numpy向量操作，for循环运行速度慢

# 复现体悟

第一次将数学知识运用起来，觉得还是挺神奇的。经典算法。

kmeans+GMM+最小割

概率论+图论

其实也可以不用kmeans，但kmeans算法比em快很多，权衡之下用的kmeans。

以后不用pycharm全面投入vsc了。vsc真的界面很清爽。

从论文理解到复现一共花了三天，本来要用c++的，但配环境花了很久最后虽然配好了，但兴趣索然全失，毕竟opencv就是C++写的，再把code重敲一遍意义不大。

但C++还是挺重要的，它运行速度很快，有空再重拾一下C++，尽管用C++刷力扣，但用C++做项目的经历还是很少，第一次下载Clion这个ide，第一次用msys2配环境，第一次了解Cmake。怀疑以前学了个假C++课。

做这个的时候面临两个期中考试，一次非限课ppt汇报，扁桃体发炎+咳嗽+角膜炎，xs人有时很脆弱有时又很强大。

不管结果如何，就当一次切磋吧，我也是有收获的。人生也只是一场没有终点的修行罢了。

最后感谢chatgpt的帮助！



# 推荐博客

[opencv源码](https://www.cnblogs.com/P3nguin/p/8532206.html)

[csdn](https://blog.csdn.net/zouxy09/article/details/8534954?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168187748416800188549457%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=168187748416800188549457&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-8534954-null-null.blog_rank_default&utm_term=grabcut&spm=1018.2226.3001.4450)

