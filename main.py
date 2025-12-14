import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import time
import numpy as np

# ==========================================
# 0. 全局配置与路径
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(CURRENT_DIR, "data", "FreiHAND_pub_v2")

BATCH_SIZE = 64
EPOCHS = 15       # 测试用，测试通过后改为 10
LEARNING_RATE = 1e-4

SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],       # 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],       # 食指
    [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
    [0, 13], [13, 14], [14, 15], [15, 16],# 无名指
    [0, 17], [17, 18], [18, 19], [19, 20] # 小指
]

# ==========================================
# 1. 网络架构
# ==========================================
class HandObjectPoseNet(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V1):
        super(HandObjectPoseNet, self).__init__()
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
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
# 2. 数据集加载器
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
            
        print("正在加载标注数据 (JSON)...")
        with open(self.xyz_path, 'r') as f:
            self.all_joints = json.load(f)
            
        with open(self.k_path, 'r') as f:
            self.all_ks = json.load(f)
            
        # === 测试模式：仅取前 500 个标注 ===
        # TEST_NUM = 500
        # print(f"!!! [测试模式] 仅截取前 {TEST_NUM} 个标注进行试验 !!!")
        # self.all_joints = self.all_joints[:TEST_NUM]
        # self.all_ks = self.all_ks[:TEST_NUM]
        # ================================
        
        self.num_samples = len(self.all_joints) * 4 
        print(f"扩增后图片总数: {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = f"{idx:08d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        
        ann_idx = idx % len(self.all_joints)
        
        joints = torch.tensor(self.all_joints[ann_idx], dtype=torch.float32)
        K = torch.tensor(self.all_ks[ann_idx], dtype=torch.float32)
        
        root_joint = joints[9].clone() 
        joints = joints - root_joint

        if self.transform:
            image = self.transform(image)
            
        return image, joints, K, root_joint

# ==========================================
# 3. 辅助函数
# ==========================================
def calculate_metrics(pred, gt, threshold=0.05):
    error = torch.norm(pred - gt, dim=2)
    mpjpe = error.mean().item()
    correct_joints = (error < threshold).sum().item()
    accuracy = correct_joints / error.numel()
    return mpjpe, accuracy

def project_points(xyz, K):
    uv = torch.matmul(K, xyz.t()).t()
    return uv[:, :2] / uv[:, 2:3]

def visualize_sample(image, pred, gt, K, root, epoch, sample_idx):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # image一般不需要grad，所以这里直接 cpu().numpy() 没问题
    img = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    
    # 【修复关键点】：pred 是模型输出，带有梯度，必须先 .detach() 再转 numpy
    pred_abs = pred.cpu().detach() + root.cpu() 
    gt_abs = gt.cpu() + root.cpu()
    K = K.cpu()
    
    # 投影后也要确保没有梯度
    p2d = project_points(pred_abs, K).numpy()
    g2d = project_points(gt_abs, K).numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    for link in SKELETON:
        plt.plot(g2d[link, 0], g2d[link, 1], 'lime', alpha=0.7, lw=2)
        plt.plot(p2d[link, 0], p2d[link, 1], 'red', alpha=0.7, lw=2)
    
    plt.axis('off')
    plt.title(f"Test Sample - Epoch {epoch}")
    plt.savefig(f'./results/epoch_{epoch}_test_{sample_idx}.png')
    plt.close()

def plot_curves(train_losses, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, 'g-', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('./results/training_metrics.png')
    print("训练指标曲线已保存至 ./results/training_metrics.png")

# ==========================================
# 4. 主训练程序
# ==========================================
def main():
    if not os.path.exists('./results'):
        os.makedirs('./results')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"硬件设备: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        full_dataset = FreiHANDDataset(DATASET_ROOT, transform=transform)
    except Exception as e:
        print(f"数据加载错误: {e}")
        return

    total_size = len(full_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"数据划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
    
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,      # 尝试设为 8，利用多核 CPU
        pin_memory=True,    # 开启内存锁页加速
        persistent_workers=True # 保持子进程存活，减少每个Epoch的启动开销
    )

    # val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,      # 验证集不需要太高
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    model = HandObjectPoseNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print("\n=== 开始计时 ===")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        
        for i, (imgs, gt, _, _) in enumerate(train_loader):
            imgs, gt = imgs.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch+1}][Train] Step {i}/{len(train_loader)} Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss_sum / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_mpjpe_sum = 0.0
        val_acc_sum = 0.0
        
        with torch.no_grad():
            for imgs, gt, _, _ in val_loader:
                imgs, gt = imgs.to(device), gt.to(device)
                pred = model(imgs)
                
                loss = criterion(pred, gt)
                val_loss_sum += loss.item()
                
                mpjpe, acc = calculate_metrics(pred, gt, threshold=0.05)
                val_mpjpe_sum += mpjpe
                val_acc_sum += acc
                
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_mpjpe = val_mpjpe_sum / len(val_loader)
        avg_val_acc = val_acc_sum / len(val_loader)
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"\n>>> Epoch {epoch+1} 完成")
        print(f"    Train Loss: {avg_train_loss:.6f}")
        print(f"    Val Loss  : {avg_val_loss:.6f}")
        print(f"    Val MPJPE : {avg_val_mpjpe:.6f}")
        print(f"    Val Acc   : {avg_val_acc*100:.2f}%")
        
        # --- 可视化 ---
        # 修正：在 infer 阶段也可以加上 no_grad，或者在内部 detach
        sample = next(iter(test_loader))
        s_img, s_gt, s_K, s_root = sample
        s_img = s_img.to(device)
        s_pred = model(s_img) # 这里出来的 pred 带有梯度
        
        # 调用 visualize_sample 时内部已经加了 .detach()
        visualize_sample(s_img[0], s_pred[0], s_gt[0], s_K[0], s_root[0], epoch+1, 0)

    end_time = time.time()
    total_time = end_time - start_time
    
    print("="*40)
    print(f"全部训练完成！")
    print(f"总耗时: {total_time/60:.2f} 分钟")
    print("="*40)
    
    plot_curves(history['train_loss'], history['val_loss'], history['val_acc'])
    # torch.save(model.state_dict(), "resnet_freihand_test.pth")
    torch.save(model.state_dict(), "resnet_freihand_full.pth")
    print("模型已保存。")

if __name__ == "__main__":
    main()