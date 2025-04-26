import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = './data' # You can change this path


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Duplicate grayscale channel to RGB
    transforms.Normalize((0.5,), (0.5,))
])

Mnist_Dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
#sample_image, _ = Mnist_Dataset[0]
dataloader = torch.utils.data.DataLoader(Mnist_Dataset, batch_size=32, shuffle=True)

latent_dim= 100
lr = 0.0002
beta1= 0.5
beta2= 0.999
num_epochs = 10

class Generator(nn.Module):
  def __init__(self, latent_dim):
    super(Generator, self).__init__()

    self.model = nn.Sequential(
        nn.Linear(latent_dim, 128*8*8),
        nn.ReLU(),
        nn.Unflatten(1, (128, 8, 8)),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128,128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128, momentum=0.78),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64, momentum=0.78),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=3, padding=1),
        nn.Tanh()
    )


  def forward(self, z):
    img= self.model(z)
    return img
  
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.BatchNorm2d(64, momentum=0.82),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128, momentum=0.82),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, momentum=0.8),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(256 * 5 * 5, 1),
        nn.Sigmoid()
    )

  def forward(self, img):
    validity = self.model(img)
    return validity
  

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()
# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

import torch
import torch.nn as nn

# Discriminator 1 (your original)
class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.82),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# Discriminator 2 (modified version)
class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(96, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(192, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),  # stays 4x4
            nn.BatchNorm2d(384, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Flatten(),

            # Output is [batch, 384 * 4 * 4] = [batch, 6144]
            nn.Linear(384 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


#Discriminator 3
class Discriminator3(nn.Module):
    def __init__(self):
        super(Discriminator3, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),  # Larger kernel size
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),  # Deeper layers
            nn.BatchNorm2d(512, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1),  # Adjusted for feature map size
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model[:-3](img)  # Process up to Conv layers before Flatten
        #print("Feature map shape before flattening:", out.shape)
        validity = self.model[-3:](out.view(out.size(0), -1))  # Flatten and pass through FC layer
        return validity

class Discriminator4(nn.Module):
    def __init__(self):
        super(Discriminator4, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=2),  # Larger kernel
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # Different kernel size
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Deeper architecture
            nn.BatchNorm2d(512, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # Extra layer for complexity
            nn.BatchNorm2d(1024, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1),  # Adjusted for feature map size
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model[:-3](img)  # Process up to Conv layers before Flatten
        validity = self.model[-3:](out.view(out.size(0), -1))  # Flatten and pass through FC layer
        return validity

class Discriminator5(nn.Module):
    def __init__(self):
        super(Discriminator5, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # Smaller initial channels, larger kernel
            nn.LeakyReLU(0.1),  # Shallower slope for LeakyReLU
            nn.Dropout(0.2),  # Lower dropout rate
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64, momentum=0.9),  # Higher momentum
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Additional layer
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),  # Adjusted for feature map size
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model[:-3](img)  # Process up to Conv layers before Flatten
        validity = self.model[-3:](out.view(out.size(0), -1))  # Flatten and pass through FC layer
        return validity
    

discriminator1 = Discriminator1().to(device)
discriminator2 = Discriminator2().to(device)
discriminator3 = Discriminator3().to(device)
discriminator4 = Discriminator4().to(device)
discriminator5 = Discriminator5().to(device)
optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D3 = torch.optim.Adam(discriminator3.parameters(),lr=lr, betas=(beta1, beta2))
optimizer_D4 = torch.optim.Adam(discriminator4.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D5 = torch.optim.Adam(discriminator5.parameters(), lr=lr, betas=(beta1, beta2))

"""def ClickConnect():
        print("Clicked on connect button")
        document.querySelector("#top-bar > div.top-bar-content > div.top-bar-right > div.top-bar-actions > div.top-bar-actions-right > div.top-bar-actions-right-items > a.top-bar-button.top-bar-button-connect").click()
setInterval(ClickConnect, 60000)"""
generator_losses = []
discriminator_losses = []
starttime = time.time()
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        #real_images = batch[0]  # First item in batch is the image tensor
        #print(real_images.shape)  # Expected: [batch_size, 3, H, W] for RGB, [batch_size, 1, H, W] for grayscale

        # Convert list to tensor
        real_images = batch[0].to(device)

        # Adversarial ground truths
        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)

        # Configure input
        real_images = real_images.to(device)

        # Sample noise as generator input
        z = torch.randn(real_images.size(0), latent_dim, device=device)
        # Generate a batch of images
        fake_images = generator(z)


        # ---------------------
        #  Train Discriminator 2
        # ---------------------
        #print(f"Real images shape: {real_images.shape}")
        optimizer_D2.zero_grad()

        real_loss_D2 = adversarial_loss(discriminator2(real_images), valid)
        fake_loss_D2 = adversarial_loss(discriminator2(fake_images.detach()), fake)
        d2_loss = (real_loss_D2 + fake_loss_D2) / 2

        d2_loss.backward()
        optimizer_D2.step()

        # ---------------------
        #  Train Discriminator 1
        # ---------------------
        optimizer_D1.zero_grad()

        real_loss_D1 = adversarial_loss(discriminator1(real_images), valid)
        fake_loss_D1 = adversarial_loss(discriminator1(fake_images.detach()), fake)
        d1_loss = (real_loss_D1 + fake_loss_D1) / 2

        d1_loss.backward()
        optimizer_D1.step()

        # ---------------------
        #  Train Discriminator 3
        # ---------------------
        optimizer_D3.zero_grad()

        real_loss_D3 = adversarial_loss(discriminator3(real_images), valid)
        fake_loss_D3 = adversarial_loss(discriminator3(fake_images.detach()), fake)
        d3_loss = (real_loss_D3 + fake_loss_D3) / 2

        d3_loss.backward()
        optimizer_D3.step()

        #Discriminator 4
        optimizer_D4.zero_grad()

        real_loss_D4 = adversarial_loss(discriminator4(real_images), valid)
        fake_loss_D4 = adversarial_loss(discriminator4(fake_images.detach()), fake)
        d4_loss = (real_loss_D4 + fake_loss_D4) / 2

        d4_loss.backward()


        # ---------------------
        #  Train Discriminator 5
        # ---------------------
        """optimizer_D5.zero_grad()

        real_loss_D5 = adversarial_loss(discriminator5(real_images), valid)
        fake_loss_D5 = adversarial_loss(discriminator5(fake_images.detach()), fake)
        d5_loss = (real_loss_D5 + fake_loss_D5) / 2

        d5_loss.backward()
        optimizer_D5.step()"""


        #  ---------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_images = generator(z)

        # Combine losses from both discriminators
        g_loss_D1 = adversarial_loss(discriminator1(gen_images), valid)
        g_loss_D2 = adversarial_loss(discriminator2(gen_images), valid)
        g_loss_D3 = adversarial_loss(discriminator3(gen_images), valid)
        g_loss_D4 = adversarial_loss(discriminator4(gen_images), valid)
        #g_loss_D5 = adversarial_loss(discriminator5(gen_images), valid)
        g_loss = (g_loss_D1 + g_loss_D2 + g_loss_D3 + g_loss_D4 ) / 4  # Average the losses
        d_loss_total = (d1_loss + d2_loss + d3_loss + d4_loss ) / 4

        discriminator_losses.append(d_loss_total)
        generator_losses.append(g_loss)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Progress Monitoring
        # ---------------------
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(dataloader)} "
                f"D1 Loss: {d1_loss.item():.4f} "
                f"D2 Loss: {d2_loss.item():.4f} "
                f"D3 Loss: {d3_loss.item():.4f} "
                f"D4 Loss: {d4_loss.item():.4f} "
                #f"D5 Loss: {d5_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )

    # Save generated images for every 10th epoch
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()

endtime = time.time()
generator_losses_np = [loss.detach().cpu().numpy() for loss in generator_losses]
discriminator_losses = [loss.detach().cpu().numpy() for loss in discriminator_losses]
print(starttime-endtime)
plt.figure(figsize=(10, 5))
plt.plot(generator_losses_np, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator vs Discriminator Loss")
plt.legend()
plt.grid(True)
plt.show()