import torch
import torch.nn as nn
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from asl_mlp import ASL_MLP


# Load dataset (MediaPipe version - 63 features: x,y,z for 21 landmarks)
data = pd.read_csv("asl_dataset_mediapipe.csv")

X = data.iloc[:, :63].values
y = data["label"].values


# Convert letters to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save the encoder so the inference script can map predictions back to letters
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)


# Create model (num_classes should match how many unique letters are in your data)
num_classes = len(encoder.classes_)
model = ASL_MLP(in_dim=63, num_classes=num_classes)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
for epoch in range(100):

    total_loss = 0

    for x_batch, y_batch in train_loader:

        optimizer.zero_grad()

        output = model(x_batch)

        loss = loss_function(output, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    if epoch % 10 == 0:
        print(
            "Epoch:",
            epoch,
            "Loss:",
            total_loss
        )


# Test accuracy
with torch.no_grad():

    output = model(X_test)

    predictions = torch.argmax(output, dim=1)

    accuracy = (predictions == y_test).float().mean()

    print("Accuracy:", accuracy.item())


# Save model
torch.save(
    model.state_dict(),
    "asl_model.pth"
)

print("Saved asl_model.pth")
print("Classes:", list(encoder.classes_))