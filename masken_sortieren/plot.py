import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy
import pandas
from PIL import Image
import cv2
import os
import shutil

"""mAP und mR aus den Metrics einzelner batches einer Epoche
von Hand berechnen um zu prüfen ob diese mit den Werten aus der summary() übereinstimmen: """
"""metrics_batches = pd.read_csv(r"C:\BA_Projekt\Project\precision_recall_bbox.csv")

mean_metrics = metrics_batches.mean()
print(mean_metrics)"""




#csv_path = r"C:\BA_Projekt\Project\precision_recall_bbox300_50_5_cd.csv"
csv_path = r"C:\BA_Projekt\Project\Losses\losses_500_5_10th.csv"
csv_df = pd.read_csv(csv_path)
csv_df = csv_df.iloc[:,3:]
csv_df.plot()
plt.xticks(np.arange(0,csv_df.shape[0],10))
plt.title("metrics")
plt.xlabel("batches")
plt.ylabel("prec/recall")

plt.show()

"""
plt.subplots(1,2)
plt.subplot(1,2,1)
a = Image.open(r"F:\Project\DropletsMask\masks\img102 Full flatfield-correct.tif")
plt.title("mask")
#a = Image.open(os.path.join("raw", folder_all_imgs[103]))
plt.imshow(a)

plt.subplot(1,2,2)
b = Image.open(r"F:\Project\DropletsMask\raw\img138 Full flatfield-correct.tif")
plt.title("original")
#b = Image.open(os.path.join("masks", folder_all_imgs[103]))
plt.imshow(b)
plt.show()

"""