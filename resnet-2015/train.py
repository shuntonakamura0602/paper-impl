import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# model.pyからResNetとBasicBlockをインポート
from model import ResNet, BasicBlock

def train():
    # --- 1. デバイスの設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. データセットとデータローダーの準備 ---
    # CIFAR-10の標準的な正規化パラメータ
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 学習用データ
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    # テスト用データ
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    # --- 3. モデル、損失関数、オプティマイザの定義 ---
    # ResNet-18をCIFAR-10用に設定
    # layers=[2, 2, 2, 2] はResNet-18を意味する
    # num_classes=10 はCIFAR-10のクラス数
    model = ResNet(BasicBlock, layers=[2, 2, 2, 2], num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 4. 学習ループ ---
    num_epochs = 10 # エポック数を10に設定
    for epoch in range(num_epochs):
        model.train() # モデルを学習モードに設定
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # データをデバイスに送る
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配をリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 逆伝播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Average Loss: {running_loss / len(train_loader):.4f}")

    print("Finished Training")

    # --- 5. 評価ループ ---
    model.eval() # モデルを評価モードに設定
    correct = 0
    total = 0
    with torch.no_grad(): # 勾配計算を無効化
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f} %")

if __name__ == "__main__":
    train()