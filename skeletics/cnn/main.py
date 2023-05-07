import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR

from cnn import CNN3D
from mydataset import MyDataset

from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
lr = .01
batch_size = 32
num_epochs = 100

# Load data
dataset = MyDataset('../parsed_mimetics/gcn_train_data.npy', '../parsed_mimetics/gcn_train_label.npy')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model and optimizer
model = CNN3D(in_channels=3, num_classes=50)
model = model.to(device)
print(device)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Define loss function
criterion = nn.CrossEntropyLoss(reduction='sum')
criterion.to(device)

# Training loop
losses, taccs, accs = [], [], []
for epoch in tqdm(range(num_epochs)):
    # Train for one epoch
    model.train()
    train_loss = 0
    num_done = 0
    for inputs, labels in train_dataloader:
    # for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}: train"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        num_done += labels.size(0)
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_loss /= num_done
    losses.append(train_loss)

    # Evaluate on test set
    model.eval()
    test_accuracy = 0
    num_samples = 0
    ps_te, ls_te = list(), list()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
        # for inputs, labels in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}: test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # currently outputs is in shape (batch_size, num_classes)
            # calculate accuracy
            # print("train")
            # print(outputs.data[0])
            # print(labels[0])
            # print()
            _, preds = torch.max(outputs.data, 1)
            ps_te.append(preds.cpu().detach().numpy())
            ls_te.append(labels.cpu().detach().numpy())
            test_accuracy += (preds == labels).sum().float().item()
            num_samples += labels.size(0)
        test_accuracy /= num_samples
        accs.append(100*test_accuracy)
    train_accuracy = 0
    num_samples = 0
    ps_tr, ls_tr = list(), list()
    with torch.no_grad():
        for inputs, labels in train_dataloader:
        # for inputs, labels in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}: test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # train_accuracy += (outputs.argmax(dim=-1) == labels).sum().float().item()
            # print("eval")
            # print(outputs.data[0])
            # print(labels[0])
            # print()
            _, preds = torch.max(outputs.data, 1)
            ps_tr.append(preds.cpu().detach().numpy())
            ls_tr.append(labels.cpu().detach().numpy())
            train_accuracy += (preds == labels).sum().float().item()
            num_samples += labels.size(0)
        train_accuracy /= num_samples
        taccs.append(100*train_accuracy)
        
    # print(losses[-1], taccs[-1], accs[-1])
    # Print results for this epoch
    # print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss:.4f}, train_accuracy = {100*train_accuracy:.2f}%, test_accuracy = {100*test_accuracy:.2f}%")

print("losses")
print(losses)
print()
print("train accs")
print(taccs)
print()
print("test accs")
print(accs)

# confusion matrix

ps_tr = np.asanyarray(np.concatenate(ps_tr))
ls_tr = np.asanyarray(np.concatenate(ls_tr))
ps_te = np.asanyarray(np.concatenate(ps_te))
ls_te = np.asanyarray(np.concatenate(ls_te))
# print(ps_tr.shape, ps_te.shape, ls_tr.shape, ls_te.shape)
# np.savez("pls.npz", ps_tr=ps_tr, ls_tr=ls_tr, ps_te=ps_te, ls_te=ls_te)

# save to model
torch.save(model.state_dict(), "model_cnn.pt")
