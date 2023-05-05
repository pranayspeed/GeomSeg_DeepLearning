











#torch segmentation CNN model using pointclouds

from GeometrySegmentation import PointCloudSegmentation, PointCloudDataset, PointNetSegmentation

import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable



# Create the training dataset

bagsfit_files_path = "/home/pranayspeed/Downloads/TRAIN-20s/"
train_dataset = PointCloudDataset(bagsfit_files_path)

batch_size=1

# Create the data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


number_of_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model = PointCloudSegmentation(num_shapes=number_of_classes)
model = model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    print("Epoch: ", epoch)
    # Iterate over the training data
    for input_points, labels in train_loader:
        
        input_points = input_points.view(batch_size, 3, -1)

        input_points = input_points.to(device)

        labels = labels.to(device).type(torch.FloatTensor)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_points).type(torch.FloatTensor)
        # print("labels[0]",  labels[0][220])
        # print("labels.shape",  labels.view(-1).shape, labels.dtype)
        # print("outputs.shape",  outputs.view(-1).shape, outputs.dtype)
        loss = criterion(outputs.view(-1), labels.view(-1))

        loss = Variable(loss, requires_grad = True)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Print the average loss for this epoch
    print("Epoch {}: Loss = {}".format(epoch, loss.item()))
    # Save the model
    torch.save(model.state_dict(), "point_cloud_segmentation.pt")

# Save the model
torch.save(model.state_dict(), "point_cloud_segmentation.pt")