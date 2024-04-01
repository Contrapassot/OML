import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import dataLoader
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO initialize weights and biases
# TODO Stopping criterion

STOPPING_CRITERION = 1e-3
STOPPING_CRITERION_EPOCHS = 5

class MLP_1(nn.Module):
    def __init__(self):
        super(MLP_1, self).__init__()
        self.seq = nn.Sequential(nn.Linear(3072, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 10), nn.BatchNorm1d(10, affine=False))  # No softmax here ?? TODO see logits instead of one-hot encoding
  
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, x):
       x = self.seq(x)
       return x
   

def get_model(name):
    if name == "MLP_1":
        return MLP_1().to(device)   

def learn(model_name, batch_size, learning_rate):
    model = get_model(model_name)
    model = model.type(torch.float32)
    model.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    
    train_dataset, test_dataset = dataLoader.load_images_labels(5)
    
    print(model)
    print(train_dataset[0][0][0])
    

    num_epochs = 1_000
    loss_values = []
    test_accuracy = []
    train_accuracy = []
    test_loss = 0
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    stopping_criterion_counter = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        if epoch == num_epochs - 1:
            print("Final epoch reached, did not converge.") # TODO throw error
            break
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, labels)
            loss.backward()
            model.optimizer.step()
            
            
            epoch_loss += loss.item()
            num_batches = i + 1
            
        loss_values.append(epoch_loss/num_batches)
        
        if epoch > 1:
            if abs(loss_values[-1] - loss_values[-2]) < STOPPING_CRITERION:
                stopping_criterion_counter += 1
        else:
            stopping_criterion_counter = 0
            
        if stopping_criterion_counter == STOPPING_CRITERION_EPOCHS:
            print(f"Stopping criterion reached at epoch {epoch} with loss difference {abs(loss_values[-1] - loss_values[-2])}")
            break
            
       
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
    learn(MLP_1, 128, 1e-4)