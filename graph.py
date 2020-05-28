# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:21:32 2019

@author: sidhant
"""

import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_excel("C:\sidhant\CA3\graph.xlsx")
data=data.dropna(axis=0)
print(data)

data.plot.bar(x='Model',y=['Accuracy','Recall','F1'], title='Model performance')