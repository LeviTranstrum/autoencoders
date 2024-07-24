import torch
import os
import json
import scipy
import numpy
from generator import Generator

class MatrixSet(torch.utils.data.Dataset):
    def __init__(self):
        data_path = './data/matrices.json'
        if not os.path.exists(data_path):
            print('generating data...')
            Generator.generate_data()

        with open(data_path) as file:
            print('loading data...')
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.flatten(torch.tensor(self.data[idx], dtype=torch.float32))

class MtoE3(torch.nn.Module):
    def __init__(self):
        super(MtoE3, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(9, 7),
            torch.nn.BatchNorm1d(7),
            torch.nn.ReLU(),
            torch.nn.Linear(7, 5),
            torch.nn.BatchNorm1d(5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 3),
            torch.nn.BatchNorm1d(3),
            torch.nn.ReLU()
        )
        self.decode = torch.nn.Sequential(
            torch.nn.Linear(3, 5),
            torch.nn.BatchNorm1d(5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 7),
            torch.nn.BatchNorm1d(7),
            torch.nn.ReLU(),
            torch.nn.Linear(7, 9),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.lossfunc = torch.nn.MSELoss()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    # Computes the angle of rotation between two rotation matrices.
    def angular_loss(self, R, R_hat):
        # Compute the relative rotation matrix R2 * R1^(-1)
        relative_rotation = torch.matmul(R_hat, R.transpose(1, 2))

        # Extract the rotation angle from the relative rotation matrix
        # The trace method is used for calculating the cosine of the rotation angle
        # cos(theta) = (trace(R) - 1) / 2
        trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1) / 2

        # Clamp the cos(theta) values for numerical stability
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # Calculate the angle in radians
        theta = torch.acos(cos_theta)

        # The loss is the mean of these angles (or you might sum them, depending on your application)
        loss = theta.sum()

    def angular_loss(self, R, R_hat):
        # Compute the relative rotation matrix R2 * R1^(-1)
        relative_rotation = torch.matmul(R_hat, R.transpose(1, 2))
        print (f'relative_rotation: {relative_rotation}')

        # Extract the rotation angle from the relative rotation matrix
        # The trace method is used for calculating the cosine of the rotation angle
        trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1) / 2
        print(f'cos_theta: {cos_theta}')

        # Clamp the cos(theta) values for numerical stability
        cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
        print(f'cos_theta_clamped: {cos_theta_clamped}')

        # Calculate the angle in radians
        theta = torch.acos(cos_theta_clamped)
        print(f'theta: {theta}')

        # The loss is the sum of these angles
        loss = theta.sum()
        print(f'loss: {loss}')

        # Check if the loss is NaN and print debugging information if so
        if torch.isnan(loss):
            print("NaN Detected!")
            print("R:", R)
            print("R_hat:", R_hat)
            print("Relative Rotation:", relative_rotation)
            print("Trace:", trace)
            print("cos_theta (before clamping):", cos_theta)
            print("cos_theta_clamped:", cos_theta_clamped)
            print("Theta:", theta)
            print("Loss:", loss)
            raise ValueError("caught a nan")

        return loss

    def frobenius_loss(self, R, R_hat):
        # Compute the Frobenius norm of the difference between R and R_hat
        loss = torch.norm(R_hat - R, p='fro')
        return loss

    def wrap_tensor_angles(self, t):
        return torch.remainder(t, 2 * torch.pi)

    def run_training(self, num_epochs=1):
        dataset = MatrixSet()
        dataloader=torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        running_loss = 0.0

        for n in range(num_epochs):
            self.train()
            running_loss = 0
            for i, data in enumerate(dataloader):
                # print(f'input: {data}')

                # zero gradients for every batch
                self.optimizer.zero_grad()

                # make predictions for this batch
                outputs = self(data)

                # print(f'outputs: {outputs}')

                # calculate loss and gradients
                loss = self.lossfunc(data.reshape(data.size(0), 3, 3), outputs.reshape(outputs.size(0), 3, 3))
                loss.backward()

                # adjust learning weights
                self.optimizer.step()

                # track loss
                running_loss += loss.item()

                # log
                if i % 100 == 99:
                    print(f'epoch {n+1} batch {i+1} average loss: {running_loss / 100}')
                    running_loss = 0.0

            self.save()
        print()

    def evaluate(self, R):
        self.eval()
        with torch.no_grad():

            R_tensor = torch.tensor(R, dtype=torch.float32).unsqueeze(0)
            print(f'initial value: {R_tensor}')
            encoded = self.encode(torch.flatten(R_tensor).unsqueeze(0))
            print(f'encoded value: {encoded}')
            decoded = self.decode(encoded).reshape(3,3)
            print(f'decoded value: {decoded}')

    def save(self):
        path = './trained_models/MtoE3'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), f'{path}')

    @classmethod
    def load(cls):
        print("loading model...")
        path = './trained_models/MtoE3'
        model = MtoE3()
        try:
            model.load_state_dict(torch.load(path))
        except:
            print(f'No trained model found with the name "MtoE3". Creating a new model')
        return model