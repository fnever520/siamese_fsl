import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchsummary import summary

class myDatasetName(Dataset):
    '''
    custom siamese dataset
    Input: 
        data: list(image_path, id)
        transforms: image transforms
    Return:
        image1
        image2
        label
    '''
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms =transforms

    def __getitem__(self,index):
        record1 = random.choice(self.data)
        # Decide randomly if return a pair of images of the same class or not
        same = random.randint(0,1)
        if same:
            while True:
                record2 = random.choice(self.data)
                if record1[1] = record2[1]:
                    label =1
                    break
        else:
            while True:
                record2 = random.choice(self.data)
                if record1[1] != record2[1]:
                    label = 0 
                    break

        # Read images
        img1 = Image.open(record1[0])
        img2 = Image.open(record2[0])

        # Convert label to tensor
        label = torch.from_numpy(np.array([label], dtype = np.float32))

        #Apply transform
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.data)

class siameseCNN(nn.module):
    """
    it returns two feature vectors to find the similarity between inputs
    Input:
        img1, img2: 105x105 grayscale image
    """
    def __init__(self):
        super().__init__()
        self.feature_maps = nn.Sequential(nn.Conv2d(1,64,10), \
                                            nn.Relu(inplace=True),\
                                            nn.MaxPool2d(2,2), \

                                            nn.Conv2d(64,128,7), \
                                            nn.Relu(inplace=True),\
                                            nn.MaxPool2d(2,2),

                                            nn.Conv2d(128,128,4),
                                            nn.Relu(inplace= True),
                                            nn.MaxPool2d(2,2),

                                            nn.Conv2d(128,256,4),
                                            nn.Relu(inplace=True))

        self.feature_vectors = nn.Sequential(nn.Linear(6*6*256, 4096),
                                            nn.Dropout(0.3),
                                            nn.Sigmoid(),

                                            nn.Linear(4096,16))

    def forward_once(self,x):
        x = self.feature_maps(x)
        x.view(x.size()[0], -1)
        x = self.feature_vectors(x)
        return x

    def forward(self, x1, x2):
        featvect1 = self.forward_once(x1)
        featvect2 = self.forward_once(x2)
        return featvect1, featvect2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = siameseCNN.to(device)
summary(model, (1,105,105))    


def contrastive_loss(featvect1, featvect2, label):
    '''
    Learn discrimitive feature for image
    '''
    margin = 2
    euclidean_distance = F.pairwise_distance(featvect1, featvect2)
    loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label)*torch.pow(torch.clamp(margin-euclidean_distance, min=0.0), 2))
    return loss

def inference(img1, img2, threshold = 0.5):
    '''
    Defines if img1 is of the same class than img2
    INput:
        img1, img2: 105 by 105 
        threshold: value between 0 and 1
    Output:
        1 if same class, 0 otherwise

    '''
    out1, out2 = siameseCNN(img1, img2)
    euclidean_distance = F.pairwise_distance(out1, out2)
    if (1-euclidean_distance.item()) > threshold:
        return 1
    return 0 