from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
# 加载日志数据
ea = event_accumulator.EventAccumulator('events.out.tfevents.1643283963.LAPTOP-HEU9D52E.4720.0')
ea.Reload()
print(ea.scalars.Keys())



metricsprecision = ea.scalars.Items('train/box_loss')
print(len(metricsprecision))
result=list()
for i in metricsprecision:
    result.append(i.value)
df=pd.DataFrame(result)
df.to_csv(r'D:\123456789.csv')