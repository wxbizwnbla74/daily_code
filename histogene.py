import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import get_data
import os
import glob
from PIL import Image
import pandas as pd 
from PIL import ImageFile
import seaborn as sns
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class ViT_10xVisium(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train,img_path,gene_path,expr_path,pos_path,barcode_path,ds=None,sr=False,aug=False):
        super(ViT_10xVisium, self).__init__()

        self.r = 224//4
        
        s=pd.read_csv(barcode_path, header=None, index_col=None)
        barcode=np.array(s[0])
        self.barcode=barcode

        gene_list=pd.read_csv(gene_path, header=None, index_col=None)
        self.gene_list=gene_list[0]

        self.train = train
        self.sr = sr
        self.aug = aug
        
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.ToTensor()
        ])
        
        names=["current"]
        
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(img_path))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(expr_path, pos_path, barcode, gene_list[0]) for i in names}

        self.exp_dict = {i:m[gene_list[0]].values for i,m in self.meta_dict.items()}
        self.center_dict = {i:m[['pxl_x','pxl_y']].values for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['pos_x','pos_y']].values for i,m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        if self.aug:
            im = self.transforms(im)
            # im = im.permute(1,2,0)
            im = im.permute(2,1,0)
        else:
            im = im.permute(1,0,2)
            # im = im

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            n_patches = len(centers)
            patches = torch.zeros((n_patches,patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()


            return patches, positions, centers

        else:    
            n_patches = len(centers)
            
            patches = torch.zeros((n_patches,patch_dim))
            exps = torch.Tensor(exps)

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

            
                if self.train:
                    return patches, positions, exps
                else: 
                    return patches, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, img_path):
        im = Image.open(img_path)
        return im

    def get_cnt(self, expr_path, barcode, gene_list):
        df = pd.read_csv(expr_path,sep=" ",header=None,index_col=None).T
        df.index=barcode
        df.columns=gene_list
        return df

    def get_pos(self, pos_path, barcode):
        pos=pd.read_csv(pos_path,header=None,index_col=0)
        pos.columns=['in_tissue','pos_x','pos_y','pxl_x','pxl_y']
        pos.index=pos.index.values
        pos=pos.loc[pos.in_tissue==1]
        pos=pos.loc[barcode]
        pos["barcode"]=pos.index

        return pos

    def get_meta(self,expr_path,pos_path,barcode, gene_list):
        cnt = self.get_cnt(expr_path,barcode, gene_list)
        pos = self.get_pos(pos_path,barcode)
        meta = cnt.join(pos.set_index('barcode'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

    
dataset = ViT_10xVisium(train=True,img_path="V1_Breast_Cancer_Block_A_Section_1_image.tif",
                        gene_path='used_gene.txt',
                        expr_path='normed_data.txt',
                        pos_path='tissue_positions_list.csv',
                        barcode_path='sample_barcode_50p.txt')

train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

   

model = HisToGene(n_layers=8, n_genes=3000, dim=1024, n_pos=128, learning_rate=1e-4)

trainer = pl.Trainer(gpus=0, max_epochs=1000)

trainer.fit(model, train_loader)

trainer.save_checkpoint("last_train.ckpt")


model = HisToGene.load_from_checkpoint("last_train.ckpt",n_layers=8, n_genes=3000, learning_rate=1e-5, n_pos=128)
device = torch.device('cpu')

dataset_test = ViT_10xVisium(train=False,img_path="V1_Breast_Cancer_Block_A_Section_1_image.tif",
                        gene_path='used_gene.txt',
                        expr_path='normed_data_all.txt',
                        pos_path='tissue_positions_list.csv',
                        barcode_path='all_barcode.txt')

test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)

adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device = device)

gene_list=pd.read_csv('used_gene.txt', header=None, index_col=None)[0]
adata_pred.var_names = gene_list
adata_truth.var_names = gene_list

s=pd.read_csv("all_barcode.txt", header=None, index_col=None)
barcode=np.array(s[0])
adata_pred.obs_names = barcode
adata_truth.obs_names = barcode

adata_pred.write_h5ad("predict.h5ad")
adata_truth.write_h5ad("raw.h5ad")
