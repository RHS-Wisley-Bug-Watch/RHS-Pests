import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import wandb

def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, test_ds, insect_class_id, other_class_id, device, threshold, epochs, results_dir):
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)
        
        # --- EVALUATION PHASE ---
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                test_loss += loss.item()
                
                probs = torch.softmax(out, dim=1)
                insect_preds = probs[:, insect_class_id] > threshold
                final_preds = torch.where(insect_preds, 
                                          torch.tensor(insect_class_id, device=device), 
                                          torch.tensor(other_class_id, device=device))
                correct += (final_preds == y).sum().item()
                total += y.size(0)
                
        test_loss = test_loss / len(test_loader)
        test_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")

        wandb.log({
            "Train Loss": train_loss,
            "Test Loss": test_loss,
            "Test Accuracy": test_acc,
            "Epoch": epoch + 1
        })

    # --- SAVE WEIGHTS ---
    weights_path = os.path.join(results_dir, "best_model.pth")
    torch.save(model.state_dict(), weights_path)
    wandb.save(weights_path)

    # --- UPLOAD VISUAL PREDICTIONS TO W&B ---
    print("\nUploading visual predictions to Weights & Biases...")
    model.eval()
    wandb_images = [] 
    test_samples = test_ds.samples
    
    with torch.no_grad():
        batch_idx = 0
        for x, y in test_loader:
            out = model(x.to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            insect_preds = (probs[:, insect_class_id] > threshold).astype(int)
            final_preds = np.where(insect_preds == 1, insect_class_id, other_class_id)
            
            for i in range(len(probs)):
                global_idx = batch_idx * test_loader.batch_size + i
                if global_idx < len(test_samples):
                    img_path, true_idx = test_samples[global_idx]
                    
                    true_name = 'Insect' if true_idx == insect_class_id else 'Other'
                    pred_name = 'Insect' if final_preds[i] == insect_class_id else 'Other'
                    insect_prob = probs[i][insect_class_id]
                    
                    try:
                        orig_img = Image.open(img_path).convert('RGB')
                        draw = ImageDraw.Draw(orig_img)
                        is_correct = (true_idx == final_preds[i])
                        border_color = "green" if is_correct else "red"
                        
                        w, h = orig_img.size
                        draw.rectangle([0, 0, w-1, h-1], outline=border_color, width=6)
                        label_text = f"True: {true_name} | Pred: {pred_name} ({insect_prob:.2f})"
                        draw.rectangle([0, 0, w, 30], fill="black")
                        draw.text((10, 8), label_text, fill="white")
                        
                        caption = "CORRECT" if is_correct else "INCORRECT"
                        wandb_images.append(wandb.Image(orig_img, caption=f"{caption}: {label_text}"))
                    except Exception:
                        pass
            batch_idx += 1

    wandb.log({"Test Set Predictions": wandb_images})
    print("Done! Check your W&B dashboard.")