import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        DATA_PATH = "cassava-leaf-disease-classification"

        train_df = pd.read_csv(DATA_PATH + "/train.csv")
        train_df.head() 

        images = train_df.image_id
        labels = train_df.label
        
        #imageExample = Image.open("cassava-leaf-disease-classification/train_images/" + images[0])
        #labelExample = labels[0]

        self.images = images
        self.labels = labels

        #print(labelExample)
        #imgplot = plt.imshow(imageExample)
        #plt.show()

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]      

        return image, label

    def __len__(self):
        return len(self.images)

