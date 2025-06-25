from torch.utils.data import DataLoader
from dataset import LeafSegmentationDataset
from transforms import transform

# Укажите верный относительный путь к папке data
root_dir = 'data'

dataset = LeafSegmentationDataset(root_dir, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)