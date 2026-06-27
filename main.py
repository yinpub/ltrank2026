import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from model.resnet import ResNetClassifier
from engine.trainer import train_one_epoch
from engine.evaluator import evaluate #,evaluatereg
from data.build import build_dataset
import os
from loss.fmmoc import FMMOC_TotalLoss
import random
from torch.amp import GradScaler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    use_amp = True  # AMP
    seed = 42
    set_seed(seed)
    if use_amp==True:
        scaler = GradScaler(enabled=use_amp)
    else:
        scaler = None
    # =========================
    # 1. Device
    # =========================
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # 2. Transform
    # =========================
    transform_train = transforms.Compose([transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1,hue=0.02,),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    transform_test = transforms.Compose([transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    # =========================
    # 3. Dataset
    # =========================
    dataset_name = "busi"
    if dataset_name == "busi":
        root_dir = '/mnt/sdb/haojie.yin/data/seg/BUSI/Dataset_BUSI_with_GT'

    train_dataset, val_dataset, test_dataset = build_dataset(dataset_name,root_dir,transform_train,transform_test)
    num_classes = train_dataset.num_classes
    class_names = train_dataset.class_names

    # 4.View data distribution
    print("Train class distribution:", np.bincount(train_dataset.labels))

    train_loader = DataLoader(train_dataset,
        batch_size=8,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset,
        batch_size=8,shuffle=False,num_workers=4)
    test_loader = DataLoader(test_dataset,
        batch_size=8,shuffle=False,num_workers=4)
    
    model = ResNetClassifier(num_classes=num_classes,pretrained=True).to(device)
    # =========================
    # 6. Loss & Optimizer
    # =========================
    # sup=False

    loss= "fmmoc"
    cls_count = np.bincount(train_dataset.labels)
    print("Class count:", cls_count)

    criterion = FMMOC_TotalLoss(n_classes=num_classes,class_counts=cls_count,device=device, gamma=0.4, delta=0.3, single_margin=False,scale=0.3)
    sup = True

    # =========================
    # 5. Model
    # =========================

    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()),lr=1e-4)
 
    # =========================
    # 7. Training Loop
    # =========================
    epochs = 80
    best_acc = 0.0

    for epoch in range(epochs):

        print(f"\n========== {loss} Epoch {epoch} ==========")

        train_loss = train_one_epoch(model,train_loader,optimizer,criterion,device,sup,scaler,use_amp)
        print(f"Train Loss: {train_loss:.4f}")
        val_score,val_acc = evaluate(model,val_loader,device,class_names=class_names,criterion=criterion,)
        
        # =========================
        # 8. Save Best Model
        # =========================

        if val_acc > best_acc:
        #if val_score < best_score:
            best_score = val_score
            best_acc = val_acc
            ckdir=f"log/{dataset_name}/{loss}/"
            os.makedirs(ckdir, exist_ok=True)
            ckpath=ckdir+"best_model.pth"
            torch.save(model.state_dict(), ckpath)
            print("Best model in {} saved.".format(epoch))
            best_epoch = epoch
    print("\nTraining finished.")
    print("Best Validation Accuracy:", best_acc)

    ckpathlast=ckdir+"last_model.pth"
    torch.save(model.state_dict(), ckpathlast)
    print("Last model in {} saved.".format(epoch))

    print("\n===== Final {} in {} Test Evaluation =====".format(loss,best_epoch))
    
    model.load_state_dict(torch.load(ckpath))
    evaluate(model,test_loader,device,class_names=test_dataset.class_names)

if __name__ == "__main__":
    main()
