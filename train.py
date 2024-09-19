import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from model import UNet3D_FeatureExtractor, LabelPredictor
from dataset import MedicalDataset
from loss import criterion, RMSELoss
from utils import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dose Prediction Model")
    
    # Dataset and Model Args
    parser.add_argument('--data_path', type=str, default='/vast/palmer/scratch/liu_xiaofeng/ss4786/dataset_nii/Train_annoymized', help='Path to the dataset')
    parser.add_argument('--ckpt_path', type=str, default='/vast/palmer/scratch/liu_xiaofeng/ss4786/ckpts', help='Path to save model weights')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (cuda/cpu)')
    
    # WandB Args
    parser.add_argument('--wandb_project', type=str, default='Dose-Prediction', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (optional)')
    parser.add_argument('--wandb_resume', action='store_true', help='Resume wandb run if available')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (optional for team tracking)')
    
    parser.add_argument('--use_wandb', type=int, default=1, help='Enable wandb logging. 1:True, 0:False')
    return parser.parse_args()

def main():
    args = parse_args()

    ckpt_path_fe = os.path.join(args.ckpt_path, "feature_extractor.pt")
    ckpt_path_lp = os.path.join(args.ckpt_path, "label_predictor.pt")
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = MedicalDataset(args.data_path)
    train_samples = int(0.8 * len(dataset))
    val_samples = int(0.2 * len(dataset))

    train_dataset = torch.utils.data.Subset(dataset, range(train_samples))
    val_dataset = torch.utils.data.Subset(dataset, range(train_samples, train_samples + val_samples))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

    # Initialize models
    feature_extractor = UNet3D_FeatureExtractor(5).to(device)
    label_predictor = LabelPredictor(256, 1).to(device)

    # Optimizers
    optimizer_fe = torch.optim.Adam(feature_extractor.parameters(), lr=args.lr)
    optimizer_lp = torch.optim.Adam(label_predictor.parameters(), lr=args.lr)

    # Load checkpoints if available
    if os.path.exists(args.ckpt_path_fe):
        feature_extractor, optimizer_fe, start_epoch, best_val_loss = load_checkpoint(ckpt_path_fe, feature_extractor, optimizer_fe, device)
        print("Loaded feature extractor checkpoint!")
    if os.path.exists(args.ckpt_path_lp):
        label_predictor, optimizer_lp, start_epoch, best_val_loss = load_checkpoint(ckpt_path_lp, label_predictor, optimizer_lp, device)
        print("Loaded label predictor checkpoint!")
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    # Initialize wandb
    if(args.use_wandb == 1):
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config={
                "learning_rate": args.lr,
                "architecture": "unet-no",
                "epochs": args.num_epochs,
                "batch_size": args.batch_size
            },
            resume=args.wandb_resume
        )
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        feature_extractor.train()
        label_predictor.train()

        train_loss = 0.0
        for inputs_dict in tqdm(train_loader):
            optimizer_fe.zero_grad()
            optimizer_lp.zero_grad()

            x4_ct, x3_ct, x2_ct, x1_ct, x3_ctv, x2_bld, x1_rct = feature_extractor(inputs_dict['ct_s'].to(device), inputs_dict['ctv'].to(device), inputs_dict['bladder'].to(device), inputs_dict['rectum'].to(device))
            denoised_imgs = label_predictor(x4_ct, x3_ct, x2_ct, x1_ct, x3_ctv, x2_bld, x1_rct)
            denoised_imgs = torch.clip(denoised_imgs, 1e-5)
            denoised_imgs = (denoised_imgs - denoised_imgs.min()) / (denoised_imgs.max() - denoised_imgs.min())

            loss = criterion(denoised_imgs, inputs_dict['ctv'].to(device), inputs_dict['bladder'].to(device), inputs_dict['rectum'].to(device), inputs_dict['applicator'].to(device), inputs_dict['dose'].to(device))
            loss.backward()

            optimizer_fe.step()
            optimizer_lp.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Training Loss: {train_loss}")    
        val_loss = 0.0

        for inputs_dict in tqdm(val_loader):
            with torch.no_grad():
                x4_ct, x3_ct, x2_ct, x1_ct, x3_ctv, x2_bld, x1_rct = feature_extractor(inputs_dict['ct_s'].to(device), inputs_dict['ctv'].to(device), inputs_dict['bladder'].to(device), inputs_dict['rectum'].to(device))
                denoised_imgs = label_predictor(x4_ct, x3_ct, x2_ct, x1_ct, x3_ctv, x2_bld, x1_rct)
            
            denoised_imgs = torch.clip(denoised_imgs, 1e-5)
            denoised_imgs = (denoised_imgs - denoised_imgs.min())/(denoised_imgs.max()-denoised_imgs.min())
            loss = criterion(denoised_imgs*10, inputs_dict['ctv'].to(device), inputs_dict['bladder'].to(device), inputs_dict['rectum'].to(device), inputs_dict['applicator'].to(device), inputs_dict['dose'].to(device)*10)
            
            val_loss += loss.item()

        n = len(val_loader)
        val_loss /= n
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Validation Loss: {val_loss}")
        
        if(args.use_wandb == 1):    
            wandb.log({"Training Loss": train_loss, "Validation Loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_fe = {
                'epoch': epoch + 1,
                'model_state_dict': feature_extractor.state_dict(),
                'optimizer_state_dict': optimizer_fe.state_dict(),
                'val_loss': best_val_loss
            }

            checkpoint_lp = {
                'epoch': epoch + 1,
                'model_state_dict': label_predictor.state_dict(),
                'optimizer_state_dict': optimizer_lp.state_dict(),
                'val_loss': best_val_loss
            }

            torch.save(checkpoint_fe, ckpt_path_fe)
            torch.save(checkpoint_lp, ckpt_path_lp)


if __name__ == "__main__":
    main()
