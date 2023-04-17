# -*- coding = utf-8 -*-
# @Time : 2023/4/14 0:22
# @Author : CQU20205644
# @File : debug4.py
# @Software : PyCharm
import maxflow

g = maxflow.Graph[int](2, 2)
nodes = g.add_nodes(2)
g.add_edge(nodes[0], nodes[1], 1, 2)
g.add_tedge(nodes[0], 2, 5)
g.add_tedge(nodes[1], 9, 4)
a=g.maxflow()
print(a)