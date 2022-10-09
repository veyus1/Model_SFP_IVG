import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy
import pandas
from PIL import Image
import cv2
import matplotlib

def plot_losses(path,n_col,label=None,batch_size=1, col_int = 1, **kwargs):
    """plots the losses of a specified csv file, the type of loss is chosen by n_col (int),
    batch_size is necessary to plot multiple losses w. different batch_sizes on the same scale"""
    df = pd.read_csv(path)
    y = np.array(df.iloc[:, n_col + 3])
    x_u = np.array(df.iloc[:, 0])
    #scale x from num_batches to num_images:
    x_s = x_u * batch_size * col_int + batch_size
    print(df.columns[n_col + 3])
    plt.plot(x_s, y, label=label, **kwargs)

def plot_metrics(path, label,n=1, n_col=0,batch_size=1, col_int = 1, **kwargs):
    """plots the metrics of a specified csv (or xlsx) file, the type of metric is chosen by n_col (int),
    batch_size is necessary to plot multiple metrics w. different batch_sizes on the same scale,
    caution: batch_size for metrics corresponds to the batch_size for the TEST set (usually 1),
    plots every nth datapoint (consider the extraction frequency from training in the .csv)"""
    ending = path.split(sep = ".")[-1]
    if ending == "csv":
        df = pd.read_csv(path).iloc[::n,:]
    elif ending == "xlsx":
        df = pd.read_excel(path).iloc[::n,:]

    y = np.array(df.iloc[:, n_col + 2])
    x_u = np.array(df.iloc[:, 0])
    #scale x from num_batches to num_images:
    x_s = x_u * batch_size * col_int + batch_size

    #label_k = df.columns[n_col + 2]
    print(df.columns[n_col + 2])
    plt.plot(x_s, y, label= f"{label}", **kwargs) # {df.columns[n_col + 2]}


"""plot_losses(path="Losses/losses_200_50_5_every_8b.csv", n_col=0, batch_size=8)
plot_losses(path="Losses/losses_200_50_5_every_4b.csv", n_col=0, batch_size=4)
plot_losses(path="Losses/losses_200_50_5_every_2b.csv", n_col=0, batch_size=2)
plt.xlabel("Bilder")
plt.ylabel("Loss")
plt.legend()
plt.show()"""

lw = 4
s = 8
c1 = (0.5,0,0)
c2 = (0.9,0.7,0.1)
c3 = (0,0, 0.5)
c4 = (0.5,0.5,0.5)
n=5

font = {'family': 'Times New Roman',
        'size': 26}
matplotlib.rc('font', **font)

"""plt.subplot(1,2,1)
plot_metrics(path=r"Metrics/precision_recall_segm_200_50_5_every_2b.csv",n=n, n_col=0, label="200 num_train",  c =c1, linewidth = lw)
plot_metrics(path=r"Metrics/precision_recall_segm_500_50_5_2_final.csv",n=n, n_col=0, label="500 num_train",  c =c2, linewidth = lw)
plt.tick_params(labelsize=20)"""

#plot_metrics(path=r"Metrics/precision_recall_segm_670_50_20_res18_3b_final.csv",n=1, n_col=8, label="mAR_200_50_5_2",  c =c1, linestyle="--")
#plot_metrics(path=r"Metrics/precision_recall_segm_200_50_5_every_4b.csv", n_col=8, label="mAP_100_50_5_2", s = s, c = c2),marker="v")

"""plt.xlabel("Evaluierungsbilder / -")
plt.ylabel("mAP / -")
#plt.title("mAR der Masken,\n über verschiedenen Batchgrößen")
plt.legend(loc = "lower right", fontsize=16)
#plt.minorticks_on()
#plt.grid(visible= True, which='both')

plt.subplot(1,2,2)
plot_metrics(path=r"Metrics/precision_recall_segm_200_50_5_every_2b.csv",n=n, n_col=8, label="200 num_train",  c =c1, linewidth = lw, ls="--")
plot_metrics(path=r"Metrics/precision_recall_segm_500_50_5_2_final.csv",n=n, n_col=8, label="500 num_train",  c =c2, linewidth = lw, ls="--")
plt.xlabel("Evaluierungsbilder / -")
plt.ylabel("mAR / -")
plt.tick_params(labelsize=20)
#plt.title("mAR der Masken,\n über verschiedenen Batchgrößen")
plt.legend(loc = "lower right", fontsize=16)
"""
"""
#plt.subplot(1,2,2)
plot_losses(path=r"Losses/losses_200_50_5_every_2b.csv", n_col=0, c =c1, batch_size=2, label="batch_size = 2",lw=lw)
plot_losses(path=r"Losses/losses_200_50_5_every_4b.csv", n_col=0,c =c2, batch_size=4, label="batch_size = 4",lw=lw)
plot_losses(path=r"Losses/losses_200_50_5_every_8b.csv", n_col=0, c =c3, batch_size=8, label="batch_size = 8",lw=lw)
plt.ylabel("Verlust / -")
plt.xlabel("Trainingsbilder / -")
#plt.title("Gesamtverlust bei verschiedener Anzahl der Trainingsbilder")
plt.legend(loc = "upper right")"""


"""


plot_metrics(path=r"Metrics/precision_recall_segm_200_50_5_every_2b.csv", n_col=8, label="resnet 50 fpn")
plot_metrics(path=r"Metrics/precision_recall_segm_200_50_5_new_bb_2b_sa.csv", n_col=8, label="resnet 18 fpn")
plt.xlabel("Eval-Bilder")
plt.ylabel("mAR")
plt.title("mAR für IoU=.50:.05:.95 der segm masks")
plt.legend(loc = "upper left")"""
#plt.minorticks_on()
#plt.grid(visible= True, which='both')

z=6
plt.figure(figsize=(15,5))
ax = plt.subplot(1,3,1)
x = torch.tensor(np.linspace(-z,z,2000))
y = torch.heaviside(x,torch.tensor(0, dtype=torch.float64))
plt.plot(x,y, label="Heaviside", lw=lw, c = "k")
plt.xlabel("x / -")
plt.ylabel("y / -")
plt.text(x=-6.2, y=0.9, s="Heaviside")
#plt.legend(loc = "upper left")
ax2 = plt.subplot(1,3,2, sharex = ax, sharey = ax)
x = torch.tensor(np.linspace(-z,z,2000))
y = torch.sigmoid(x)
plt.plot(x,y, label="Sigmoid", lw=lw, c = "k")
plt.text(x=-6.2, y=0.9, s="Sigmoid")
plt.xlabel("x / -")
plt.ylabel("y / -")

plt.subplot(1,3,3)
x = torch.tensor(np.linspace(-z,z,2000))
y = torch.relu(x)
plt.plot(x,y, label="ReLU", lw=lw, c = "k")
plt.text(x=-6.2, y=5.4, s="ReLU")
plt.xlabel("x / -")
plt.ylabel("y / -")

plt.show()
