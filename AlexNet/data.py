# External libraries
from torchvision import datasets, transforms


# Transforming data into required size and adding channels
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resized to required dimensions for AlexNet
    transforms.Grayscale(num_output_channels=3),  # change RGB channels
    transforms.ToTensor(),
])

# Downloading the test and training set in a ratio of 70/10
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)