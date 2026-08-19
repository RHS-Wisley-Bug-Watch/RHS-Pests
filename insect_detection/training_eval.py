import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, test_ds, 
                       insect_class_id, other_class_id, device, threshold, epochs, results_dir):
    
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

    print("\nUploading visual predictions and raw data to Weights & Biases...")
    model.eval()
    wandb_images = [] 
    test_samples = test_ds.samples
    
    all_true_labels = []
    all_pred_labels = []
    all_insect_probs = []
    all_other_probs = []
    all_image_types = []
    
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
                    insect_prob = float(probs[i][insect_class_id])
                    other_prob = float(probs[i][other_class_id])
                    
                    img_type = "Augmented" if "_aug_" in img_path else "Original"
                    
                    all_true_labels.append(true_name)
                    all_pred_labels.append(pred_name)
                    all_insect_probs.append(insect_prob)
                    all_other_probs.append(other_prob)
                    all_image_types.append(img_type)
                    
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
                        
                        caption = f"{'CORRECT' if is_correct else 'INCORRECT'} | Type: {img_type}"
                        wandb_images.append(wandb.Image(orig_img, caption=caption))
                    except Exception:
                        pass
            batch_idx += 1

    wandb.log({"Test Set Predictions": wandb_images})
    
    print("Uploading Probability Table...")
    prob_table = wandb.Table(columns=["True Label", "Predicted Label", "Insect Probability", "Other Probability", "Image Type"])
    for t_label, p_label, i_prob, o_prob, img_type in zip(all_true_labels, all_pred_labels, all_insect_probs, all_other_probs, all_image_types):
        prob_table.add_data(t_label, p_label, i_prob, o_prob, img_type)
        
    wandb.log({"Evaluation_Data": prob_table})
    
    print("Generating Density Plot and ROC Curve for W&B...")
    
    true_insect_scores = [prob for prob, true_lbl in zip(all_insect_probs, all_true_labels) if true_lbl == 'Insect']
    true_other_scores = [prob for prob, true_lbl in zip(all_insect_probs, all_true_labels) if true_lbl == 'Other']

    plt.figure(figsize=(8, 5))
    sns.kdeplot(true_insect_scores, color='blue', label='Actual Insects', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(true_other_scores, color='orange', label='Actual Others', fill=True, alpha=0.3, linewidth=2)
    plt.title('Confidence: Actual Insects vs. Actual Others')
    plt.xlabel('Predicted probability of being an Insect')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    wandb.log({"Probability Density": wandb.Image(plt)})
    plt.close()

    binary_true_labels = [1 if label == 'Insect' else 0 for label in all_true_labels]
    
    if len(set(binary_true_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(binary_true_labels, all_insect_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        wandb.log({"ROC Curve": wandb.Image(plt)})
        plt.close()

    print("Done! Check your W&B dashboard.")