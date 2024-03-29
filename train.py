import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import dataLoader
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP_1(nn.Module):
    def __init__(self):
        super(MLP_1, self).__init__()
        self.l1 = nn.Linear(3072, 1000) 
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 200)
        self.l4 = nn.Linear(200, 100)
        self.l5 = nn.Linear(100, 10)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
       x = F.relu(self.l1(x))
       x = F.relu(self.l2(x))
       x = F.relu(self.l3(x))
       x = F.relu(self.l4(x))
       x = F.softmax(self.l5(x))
       return x
   
   
   
   
   
   
def learn():
    model = MLP_1().to(device)
    model = model.type(torch.float32)
    train_dataset, test_dataset = dataLoader.load_images_labels(1)
    
    print(model)
    print(train_dataset[0][0][0])
    

    batch_size = 256
    num_epochs = 100
    loss_values = []
    test_accuracy = []
    train_accuracy = []
    test_loss = 0
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, labels)
            loss.backward()
            model.optimizer.step()
            
            loss_values.append(loss.item())
            
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
                
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        test_accuracy.append(100 * correct / total)
        test_loss = loss.item()
        print(f"Test accuracy: {100 * correct / total}")
    
    

if __name__ == '__main__':
    learn()