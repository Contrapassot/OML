import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import dataLoader
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO initialize weights and biases
# TODO Stopping criterion

class MLP_1(nn.Module):
    def __init__(self):
        super(MLP_1, self).__init__()
        self.seq = nn.Sequential(nn.Linear(3072, 1000) , nn.BatchNorm1d(1000), nn.ReLU(),
                                 nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(),
                                 nn.Linear(500, 200), nn.BatchNorm1d(200), nn.ReLU(),
                                 nn.Linear(200, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                 nn.Linear(100, 10), nn.BatchNorm1d(10))  # No softmax here ?? TODO see logits instead of one-hot encoding
  
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
       x = self.seq(x)
       return x
   
   
# TODO make this a function of batch size, model, learning rate, etc. 
def learn():
    model = MLP_1().to(device)
    model = model.type(torch.float32)
    train_dataset, test_dataset = dataLoader.load_images_labels(5)
    
    print(model)
    print(train_dataset[0][0][0])
    

    batch_size = 256
    num_epochs = 30
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
    
    return model
    

if __name__ == '__main__':
    learn()