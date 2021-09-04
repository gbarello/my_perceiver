import perceiver
import torch
from training_utils import get_positional_embedding, prep_data
import torchvision
from torch import optim
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from torchvision.datasets import MNIST, ImageNet
import json

dev = "cuda:1"

#Load the datasets
data = MNIST("/home/gabriel/Data/MNIST/", download = True, train = True, transform = torchvision.transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0)

testdata = MNIST("/home/gabriel/Data/MNIST/", download = True, train = False, transform = torchvision.transforms.ToTensor())
test_data_loader = torch.utils.data.DataLoader(testdata,
                                          batch_size=1024,
                                          shuffle=False,
                                          num_workers=0)


#build the postion encodings
static_pos_encoding = torch.from_numpy(get_positional_embedding([28,28],4)).float().to(dev)

base_learned_pos_encoding = torch.randn(len(static_pos_encoding),28,28,requires_grad = True)
learned_pos_encoding = base_learned_pos_encoding.to(dev)

#build the models
static_pos_model = perceiver.Perceiver(data_channels = len(static_pos_encoding) + 1,
                            latent_dimention = 8,
                            latent_channels = 16,
                            n_attends = 6, 
                            n_transform_per_attend = 3,
                            share_params = True, 
                            share_first_params = False,
                            n_heads = 2,
                            head_dim = 16,
                            dropout = .1,
                            output_dimention = 10).to(dev)
learned_pos_model = perceiver.Perceiver(data_channels = len(base_learned_pos_encoding) + 1,
                            latent_dimention = 8,
                            latent_channels = 16,
                            n_attends = 6, 
                            n_transform_per_attend = 3,
                            share_params = True, 
                            share_first_params = False,
                            n_heads = 2,
                            head_dim = 16,
                            dropout = .1,
                            output_dimention = 10).to(dev)
no_pos_model = perceiver.Perceiver(data_channels = 1,
                            latent_dimention = 8,
                            latent_channels = 16,
                            n_attends = 6, 
                            n_transform_per_attend = 3,
                            share_params = True, 
                            share_first_params = False,
                            n_heads = 2,
                            head_dim = 16,
                            dropout = .1,
                            output_dimention = 10).to(dev)

#Build the optimizer and the loss function
optimizer = optim.Adam(
    list(static_pos_model.parameters()) 
    + 
    [base_learned_pos_encoding]
    +
    list(learned_pos_model.parameters())
    + 
    list(no_pos_model.parameters()),.0002)

loss_module = nn.CrossEntropyLoss()
        
#Train!
full_loss_list = []
for k in range(300):
    print(f"Epoch: {k}")
    loss_list = []

    static_pos_model.train()
    learned_pos_model.train()
    no_pos_model.train()
    
    for data, labels in tqdm(data_loader):
        
        static_pos_data = prep_data(data.to(dev),static_pos_encoding,add_enc = True)
        learned_pos_data = prep_data(data.to(dev),learned_pos_encoding,add_enc = True)
        no_pos_data = prep_data(data.to(dev),None,add_enc = False)
        
        static_model_output = static_pos_model(static_pos_data)
        learned_model_output = learned_pos_model(learned_pos_data)
        model_output = no_pos_model(no_pos_data)

        s_loss = loss_module(static_model_output,labels.to(dev))
        l_loss = loss_module(learned_model_output,labels.to(dev))
        n_loss = loss_module(model_output,labels.to(dev))
        
        loss = s_loss + l_loss + n_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append([s_loss.item(),l_loss.item(),n_loss.item()])

        
    loss_list = np.array(loss_list)
    
    print(f"Train Loss: {np.mean(loss_list,0)}")
    
    trloss = np.mean(loss_list,0)
    
    static_pos_model.eval()
    learned_pos_model.eval()
    no_pos_model.eval()
    
    with torch.no_grad():
        loss_list = []
        for data, labels in test_data_loader:
            static_pos_data = prep_data(data.to(dev),static_pos_encoding,add_enc = True)
            learned_pos_data = prep_data(data.to(dev),learned_pos_encoding,add_enc = True)
            no_pos_data = prep_data(data.to(dev),None,add_enc = False)

            static_model_output = static_pos_model(static_pos_data)
            learned_model_output = learned_pos_model(learned_pos_data)
            model_output = no_pos_model(no_pos_data)

            s_loss = loss_module(static_model_output,labels.to(dev))
            l_loss = loss_module(learned_model_output,labels.to(dev))
            n_loss = loss_module(model_output,labels.to(dev))
            
            loss_list.append([s_loss.item(),l_loss.item(),n_loss.item()])

        loss_list = np.array(loss_list)
        print(f"Test Loss: {np.mean(loss_list,0)}")
    teloss = np.mean(loss_list,0)
    full_loss_list.append([trloss,teloss])
    
#Save the losses
full_loss_list = np.transpose(np.array(full_loss_list),[1,2,0])
losses = {label + "_" + modeltype:l.tolist() for L,label in zip(full_loss_list,["Train","Test"]) for l, modeltype in zip(L,["static_pos","learned_pos","no_pos"])}
json.dump(losses,open("./all_losses.json","w"))