import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append("../")

from bagsfit import BAGsFit_resnet101, WeightedMultiLabelSigmoidLoss, check_gpu
from GeometrySegmentation import PointCloudSegmentation, PointCloudNormalDataset, PointNetSegmentation, PointCloudDataset

from tqdm import tqdm


def main():
    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)
     
    # Arguments passed
    print("\nName of Python script:", sys.argv[0])
    
    batch_size=1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = 0 if torch.cuda.is_available() else None    
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    if n >=2:
        batch_size = int(sys.argv[1])
        if n==3:
            device_in = sys.argv[2]
            if device_in =="cuda":
                if not torch.cuda.is_available():
                    device="cpu"
                    is_cuda=None
                    device_str ="cpu"
            else:
                device="cpu"
                is_cuda=None
                device_str ="cpu"
                
    
    number_of_classes = 6
    model = BAGsFit_resnet101(number_of_classes)

    # Create the training dataset

    #bagsfit_files_path = "/home/pranayspeed/Downloads/TRAIN-20s-normals/"
    #train_dataset = PointCloudNormalDataset(bagsfit_files_path)
    bagsfit_files_path = "/home/pranayspeed/Downloads/TRAIN-20s/"
    train_dataset = PointCloudDataset(bagsfit_files_path)


    

    # Create the data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)




    
    
    print("is_cuda:",is_cuda)
    print("cuda", device)
    
    model = model.to(device)
    model.eval()
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    

    # Define the optimizer
    policies = get_model_policy(model)
    optimizer = torch.optim.SGD(policies, lr=0.1, momentum=0.9, weight_decay=5e-4)


    #print(model.is_cuda)
    # Training loop
    for epoch in range(100):
        print("Epoch: ", epoch)
        with tqdm(train_loader) as pbar:
            # Iterate over the training data
            for input_points, labels in pbar:
                perm = torch.randperm(input_points.size(1))
                idx = perm[:440*440]
                samples_input = input_points[:,idx]
                labels = labels[:,idx]
                input_points = samples_input.view(batch_size, 3, 440,-1)
                # Input for Image CNN.
                img_var = input_points.to(device)#check_gpu(is_cuda, input_points) # BS X 3 X H X W

                labels[labels==-1]=5
                target_var = labels.type(torch.LongTensor).to(device)#check_gpu(is_cuda, labels) # BS X H X W X NUM_CLASSES

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                score_feats5, fused_feats =  model(img_var)
                
                

                print("score_feats5.shape", score_feats5.shape, score_feats5.dtype)
                print("target_var.shape", target_var.shape, target_var.dtype)
                
                score_feats5, target_var = score_feats5.reshape(batch_size,number_of_classes,-1), target_var.reshape(batch_size, -1)#.type(torch.LongTensor)
                #print(score_feats5.is_cuda, target_var.is_cuda)

                #target_var = F.one_hot(target_var, num_classes=number_of_classes)        

                print("score_feats5[:20]", score_feats5[0,:20])
                #pred_var = torch.softmax(score_feats5, dim=1)
                print("target_var[:20]", target_var[0,:20])
                print("score_feats5[:20]", score_feats5[0,:20])
                #print("pred_var.shape", score_feats5.shape)
                #print("target_var.shape", target_var.shape)

                #exit(0)
                loss = criterion(score_feats5, target_var)
                print("loss", loss)

                yhat=torch.max(score_feats5.data,1)
                print("yhat", yhat[:20])
                # print("loss.shape", loss.shape)
                # print("loss.dtype", loss.dtype)

                # #loss = Variable(loss, requires_grad = True) 

                # print("loss.shape", loss[:, :20])

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            # Print the average loss for this epoch
            print("Epoch {}: Loss = {}".format(epoch, loss.item()))
            # Save the model
            torch.save(model.state_dict(), "point_cloud_segmentation_bagsfit_pointsonly_"+device_str+".pt")

    # Save the model
    torch.save(model.state_dict(), "point_cloud_segmentation_bagsfit_pointsonly_"+device_str+".pt")



def get_model_policy(model):
    score_feats_conv_weight = []
    score_feats_conv_bias = []
    other_pts = []
    for m in model.named_modules():
        if m[0] != '' and m[0] != 'module':
            if ('score' in m[0] or 'fusion' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                ps = list(m[1].parameters())
                score_feats_conv_weight.append(ps[0])
                if len(ps) == 2:
                    score_feats_conv_bias.append(ps[1])
                print("Totally new layer:{0}".format(m[0]))
            else: # For all the other module that is not totally new layer.
                ps = list(m[1].parameters())
                other_pts.extend(ps)

    return [
            {'params': score_feats_conv_weight, 'lr_mult': 10, 'name': 'score_conv_weight'},
            {'params': score_feats_conv_bias, 'lr_mult': 20, 'name': 'score_conv_bias'},
            {'params': filter(lambda p: p.requires_grad, other_pts), 'lr_mult': 1, 'name': 'other'},
    ]


if __name__ == '__main__':
    main()


