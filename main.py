import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import json
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置路径
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(CURRENT_DIR, "data", "FreiHAND_pub_v2")

# 手部骨架连接定义 (FreiHAND/MANO 标准)
# 拇指: 0-1-2-3-4, 食指: 0-5-6-7-8, ...
SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],       # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       # Index
    [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
    [0, 13], [13, 14], [14, 15], [15, 16],# Ring
    [0, 17], [17, 18], [18, 19], [19, 20] # Pinky
]

# ==========================================
# 2. 定义网络架构
# ==========================================
class HandObjectPoseNet(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V1):
        super(HandObjectPoseNet, self).__init__()
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 输出 21 * 3 = 63 维坐标
        self.hand_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 21 * 3) 
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        hand_pose = self.hand_head(features)
        hand_pose = hand_pose.view(-1, 21, 3)
        return hand_pose

# ==========================================
# 3. 数据集加载器 (升级版：读取K矩阵)
# ==========================================
class FreiHANDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.img_dir = os.path.join(root_dir, 'training', 'rgb')
        self.xyz_path = os.path.join(root_dir, 'training_xyz.json')
        self.k_path = os.path.join(root_dir, 'training_K.json')
        
        if not os.path.exists(self.xyz_path):
            raise FileNotFoundError(f"找不到标注文件: {self.xyz_path}")
            
        print("正在加载标注数据...")
        with open(self.xyz_path, 'r') as f:
            self.all_joints = json.load(f)
            
        with open(self.k_path, 'r') as f:
            self.all_ks = json.load(f)

        # 为了快速演示，只取前 500 个样本
        # 正式训练请注释掉下面这行
        self.all_joints = self.all_joints[:500]
        self.all_ks = self.all_ks[:500]
        
        print(f"加载成功，样本数: {len(self.all_joints)}")

    def __len__(self):
        return len(self.all_joints)

    def __getitem__(self, idx):
        img_name = f"{idx:08d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        # 读取数据
        joints = torch.tensor(self.all_joints[idx], dtype=torch.float32)
        K = torch.tensor(self.all_ks[idx], dtype=torch.float32)
        
        # 记录原始根节点坐标 (用于后续可视化恢复绝对深度)
        root_joint = joints[9].clone() 
        
        # 归一化：相对坐标 (减去中指根部)
        joints = joints - root_joint

        if self.transform:
            image = self.transform(image)
            
        return image, joints, K, root_joint

# ==========================================
# 4. 工具函数：可视化与绘图
# ==========================================
def project_points_3d_to_2d(xyz, K):
    """
    将3D点投影到2D图像平面
    xyz: [N, 3]
    K:   [3, 3]
    return: [N, 2]
    """
    # 矩阵乘法: (K * xyz^T)^T
    uv = torch.matmul(K, xyz.t()).t()
    # 透视除法: x' = x/z, y' = y/z
    return uv[:, :2] / uv[:, 2:3]

def visualize_results(image_tensor, pred_3d, gt_3d, K, root_joint, epoch, index):
    """
    绘制对比图并保存
    """
    # 反归一化图像 (Tensor -> Numpy RGB)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    
    # 恢复绝对坐标用于投影 (Pred_Rel + GT_Root)
    # 注意：实际预测中应该预测绝对深度，这里简化处理，假设深度已知
    pred_abs = pred_3d.cpu() + root_joint.cpu()
    gt_abs = gt_3d.cpu() + root_joint.cpu()
    K = K.cpu()
    
    # 投影到 2D
    pred_2d = project_points_3d_to_2d(pred_abs, K).numpy()
    gt_2d = project_points_3d_to_2d(gt_abs, K).numpy()
    
    # 绘图
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    # 画骨架连线
    for link in SKELETON:
        # Ground Truth (绿色)
        plt.plot(gt_2d[link, 0], gt_2d[link, 1], color='lime', linewidth=2, alpha=0.7, label='GT' if link==SKELETON[0] else "")
        # Prediction (红色)
        plt.plot(pred_2d[link, 0], pred_2d[link, 1], color='red', linewidth=2, alpha=0.7, label='Pred' if link==SKELETON[0] else "")
        
    plt.legend()
    plt.title(f"Epoch {epoch} - Sample {index}")
    plt.axis('off')
    
    # 保存图片
    if not os.path.exists('./results'):
        os.makedirs('./results')
    plt.savefig(f'./results/epoch_{epoch}_sample_{index}.png')
    plt.close()

def plot_loss_curve(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/loss_curve.png')
    print("Loss曲线已保存至 ./results/loss_curve.png")

# ==========================================
# 5. 训练主程序
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = FreiHANDDataset(DATASET_ROOT, transform=transform)
    except Exception as e:
        print(f"错误: {e}")
        return

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # num_workers=0 避免Windows报错
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    # 测试集batch_size设为1，方便可视化
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = HandObjectPoseNet(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 10  # 增加一点轮数看曲线
    loss_history = []
    
    print("\n=== 开始训练 ===")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (imgs, gt_hand, _, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            gt_hand = gt_hand.to(device)
            
            optimizer.zero_grad()
            pred_hand = model(imgs)
            loss = criterion(pred_hand, gt_hand)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Step {i}, Loss: {loss.item():.6f}")
        
        # --- 每个Epoch结束后，在测试集上随机抽取一张进行可视化 ---
        model.eval()
        with torch.no_grad():
            # 取测试集第一个样本
            sample_img, sample_gt, sample_K, sample_root = next(iter(test_loader))
            sample_img = sample_img.to(device)
            pred_hand = model(sample_img) # [1, 21, 3]
            
            # 调用可视化函数 (只画第一张图)
            visualize_results(
                sample_img[0], 
                pred_hand[0], 
                sample_gt[0], 
                sample_K[0], 
                sample_root[0], 
                epoch + 1, 
                0
            )
            model.train() # 切回训练模式

    # 训练结束，绘制Loss曲线
    plot_loss_curve(loss_history)
    
    # 保存模型
    torch.save(model.state_dict(), "resnet_freihand_final.pth")
    print("训练完成。结果图片保存在 ./results 文件夹中。")

if __name__ == "__main__":
    # 创建结果保存目录
    if not os.path.exists('./results'):
        os.makedirs('./results')
    train_model()