import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define the Expert Model
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

def forward(self, x):
    x = torch.relu(self.layer1(x))
    return torch.softmax(self.layer2(x), dim=1)

# 2. Define the Simplified Gating Network
class Gating(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gating, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)  # Single hidden layer
        self.layer2 = nn.Linear(16, num_experts)

def forward(self, x):
    x = torch.relu(self.layer1(x))
    return torch.softmax(self.layer2(x), dim=1)

# 3. Define the Mixture of Experts (MoE) Model
class MoE(nn.Module):
    def __init__(self, experts, top_k=2):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating = Gating(input_dim=experts[0].layer1.in_features, num_experts=len(experts))
        self.top_k = top_k

def forward(self, x):
    weights = self.gating(x)
    top_k_weights, top_k_indices = torch.topk(weights, k=self.top_k, dim=1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
    batch_size = x.size(0)
    output = torch.zeros(batch_size, self.experts[0].layer2.out_features, device=x.device)
    for i in range(batch_size):
        for k in range(self.top_k):
            expert_idx = top_k_indices[i, k]
            weight = top_k_weights[i, k]
            expert_output = self.experts[expert_idx](x[i:i+1])
            output[i] += weight * expert_output.squeeze(0)
    return output

# 4. Define the Single Large Neural Network
class SingleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SingleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)

def forward(self, x):
    x = torch.relu(self.layer1(x))
    x = torch.relu(self.layer2(x))
    return torch.softmax(self.layer3(x), dim=1)

# 5. Function to Count Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 6. Generate Synthetic Data
num_samples = 5000
input_dim = 4
output_dim = 3x_data = torch.randn(num_samples, input_dim)
y_data = torch.cat([torch.zeros(num_samples // 3),
                    torch.ones(num_samples // 3),
                    torch.full((num_samples - 2 * (num_samples // 3),), 2)]).long()for i in range(num_samples):
    if y_data[i] == 0:
        x_data[i, 0] += 1
    elif y_data[i] == 1:
        x_data[i, 1] -= 1
    elif y_data[i] == 2:
        x_data[i, 0] -= 1shuffled_indices = torch.randperm(num_samples)
x_data, y_data = x_data[shuffled_indices], y_data[shuffled_indices]train_size = int(0.8 * num_samples)
x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]

# 7. Train the Experts
hidden_dim = 32
epochs = 100
learning_rate = 0.001experts = [Expert(input_dim, hidden_dim, output_dim) for _ in range(3)]
optimizers = [optim.Adam(expert.parameters(), lr=learning_rate) for expert in experts]for i, expert in enumerate(experts):
    optimizer = optimizers[i]
    mask = y_train == i
    x_train_subset, y_train_subset = x_train[mask], y_train[mask]
    if len(x_train_subset) == 0:
        continue
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = expert(x_train_subset)
        loss = nn.CrossEntropyLoss()(outputs, y_train_subset)
        loss.backward()
        optimizer.step()

# 8. Train the MoE Model
moe_model = MoE(experts, top_k=2)
optimizer_moe = optim.Adam(moe_model.parameters(), lr=learning_rate)for epoch in range(epochs):
    optimizer_moe.zero_grad()
    outputs_moe = moe_model(x_train)
    loss_moe = nn.CrossEntropyLoss()(outputs_moe, y_train)
    loss_moe.backward()
    optimizer_moe.step()
    if (epoch + 1) % 20 == 0:
        print(f"MoE Epoch [{epoch+1}/{epochs}], Loss: {loss_moe.item():.4f}")
      
# 9. Train the Single Large NN (Layers Params: 4→350→190→3)
single_nn = SingleNN(input_dim=4, hidden_dim1=350, hidden_dim2=190, output_dim=3)
optimizer_nn = optim.Adam(single_nn.parameters(), lr=learning_rate)for epoch in range(epochs):
    optimizer_nn.zero_grad()
    outputs_nn = single_nn(x_train)
    loss_nn = nn.CrossEntropyLoss()(outputs_nn, y_train)
    loss_nn.backward()
    optimizer_nn.step()
    if (epoch + 1) % 20 == 0:
        print(f"Single NN Epoch [{epoch+1}/{epochs}], Loss: {loss_nn.item():.4f}")
      
# 10. Evaluate Both Models
def evaluate(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y).sum().item()
        return correct / len(y)accuracy_moe = evaluate(moe_model, x_test, y_test)
accuracy_nn = evaluate(single_nn, x_test, y_test)

# 11. Calculate Parameters
moe_total_params = count_parameters(moe_model)
moe_active_params = count_parameters(moe_model.gating) + 2 * count_parameters(experts[0])  # 2 experts active
nn_params = count_parameters(single_nn)

# 12. Print Results
print("\nResults Comparison:")
print(f"MoE Total Parameters: {moe_total_params}")
print(f"MoE Active Parameters (Inference): {moe_active_params}")
print(f"Single NN Parameters: {nn_params}")
print(f"MoE Model Accuracy: {accuracy_moe:.4f}")
print(f"Single NN Accuracy: {accuracy_nn:.4f}")

