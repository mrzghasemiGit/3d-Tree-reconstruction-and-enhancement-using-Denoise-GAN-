import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PointCloudDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.max_points = 2048

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        cloud = o3d.io.read_point_cloud(self.file_list[idx])
        points = np.asarray(cloud.points)
        
        # Normalization and padding
        points = self._normalize(points)
        points = self._adjust_size(points)
        
        return torch.tensor(points, dtype=torch.float32)

    def _normalize(self, data):
        centroid = np.mean(data, axis=0)
        data -= centroid
        max_dist = np.max(np.linalg.norm(data, axis=1))
        return data / max_dist

    def _adjust_size(self, data):
        if len(data) > self.max_points:
            return data[:self.max_points]
        return np.pad(data, ((0, self.max_points - len(data)), (0, 0)), 'constant')

class ReconstructionGenerator(nn.Module):
    def __init__(self, num_points=2048):
        super().__init__()
        self.initial_fc = nn.Linear(num_points*3, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.initial_fc(x)
        x = x.unsqueeze(2)
        return self.decoder(x).transpose(1, 2)

class StructureDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 1),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = torch.max(x, 2)[0]
        return self.classifier(x)

class DenoisingLoss(nn.Module):        
    def forward(self, generated, noisy_input, discriminator, threshold=90):
        # Structural consistency loss
        recon_loss = torch.mean(torch.norm(generated - noisy_input, dim=2)**2
        
        # Adaptive noise suppression
        noisy_input.requires_grad = True
        d_real = discriminator(noisy_input)
        gradients = torch.autograd.grad(
            outputs=d_real,
            inputs=noisy_input,
            grad_outputs=torch.ones_like(d_real),
            retain_graph=True,
            create_graph=True
        )[0]
        grad_norms = torch.norm(gradients, dim=2)
        cutoff = np.percentile(grad_norms.cpu().detach().numpy(), threshold)
        mask = (grad_norms < cutoff).float()
        feature_loss = torch.mean(mask * torch.norm(generated - noisy_input, dim=2)**2
        
        return 0.7*(1 - d_real.mean()) + 0.2*recon_loss + 0.1*feature_loss

def train_denoiser(data_loader, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = ReconstructionGenerator().to(device)
    discriminator = StructureDiscriminator().to(device)
    
    gen_opt = torch.optim.Adam(generator.parameters(), lr=0.0001)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=0.00001)
    
    criterion = DenoisingLoss()
    
    for epoch in range(num_epochs):
        for batch in data_loader:
            batch = batch.to(device)
            
            # Update discriminator
            disc_opt.zero_grad()
            
            real_pred = discriminator(batch)
            loss_real = -torch.mean(torch.log(real_pred + 1e-8))
            
            fake_data = generator(batch)
            fake_pred = discriminator(fake_data.detach())
            loss_fake = -torch.mean(torch.log(1 - fake_pred + 1e-8))
            
            disc_loss = loss_real + loss_fake
            disc_loss.backward()
            disc_opt.step()
            
            # Update generator
            gen_opt.zero_grad()
            gen_loss = criterion(fake_data, batch, discriminator)
            gen_loss.backward()
            gen_opt.step()
            
        print(f"Epoch {epoch+1}/{num_epochs} | D Loss: {disc_loss.item():.4f} | G Loss: {gen_loss.item():.4f}")
    
    return generator

def process_result(generated, source_path):
    source_cloud = o3d.io.read_point_cloud(source_path)
    source_points = np.asarray(source_cloud.points)
    
    centroid = np.mean(source_points, axis=0)
    max_span = np.max(np.linalg.norm(source_points - centroid, axis=1))
    
    generated = generated * max_span + centroid
    result_cloud = o3d.geometry.PointCloud()
    result_cloud.points = o3d.utility.Vector3dVector(generated)
    
    o3d.io.write_point_cloud("enhanced_output.ply", result_cloud)
    o3d.visualization.draw_geometries([result_cloud])
    
    return result_cloud

if __name__ == "__main__":
    # Input: Fused point clouds from previous step
    input_files = ["fused_1.ply", "fused_2.ply"]  
    dataset = PointCloudDataset(input_files)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    trained_generator = train_denoiser(loader, num_epochs=50)
    
    sample_input = dataset[0].unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        enhanced = trained_generator(sample_input).cpu().numpy()[0]
    
    process_result(enhanced, input_files[0])