import time
import numpy
import tools
import training.train as train
import training.test as test
from datetime import datetime
import torch.nn as nn

start = datetime.now()

t = int(time.time())
# t = 1665789487
modelPath = f"models/{t}"
dataPathTrain = f"numpy/training/{t}"
dataPathTest = f"numpy/testing/{t}"

print(modelPath)
print(dataPathTest)
print(dataPathTrain)


print("Formatting")


tools.format(dataPathTrain, 0, 25000, skip_ties=True)
tools.format(dataPathTest, 25000, 1000)

formatend = datetime.now()
td = (formatend - start).total_seconds() * 10**3
print(f"Format time : {td:.03f}ms")

print("Training")
train.run_train(modelPath, 
    dataPathTrain + "_x.npy",
    dataPathTrain + "_y.npy",
    30, 
    useCUDA=True,
    batchSize=10)

trainend = datetime.now()

td = (trainend - formatend).total_seconds() * 10**3
print(f"Training time : {td:.03f}ms")


# print("Testing")
# train.run_train(modelPath, 
#     dataPathTrain + "_x.npy",
#     dataPathTrain + "_y.npy",
#     10)

end = datetime.now()
td = (end - trainend).total_seconds() * 10**3
print(f"Testing time : {td:.03f}ms")

