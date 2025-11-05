import numpy as np
import pandas as pd
import torch
import os
import time
from utils.utils import *
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")  # Debugging: Print device info


class Accuracy_Logger(object):
    """Tracks accuracy for each class."""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat, Y = int(Y_hat), int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat, Y = np.array(Y_hat).astype(int), np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count, correct = self.data[c]["count"], self.data[c]["correct"]
        return (None if count == 0 else float(correct) / count, correct, count)

class EarlyStopping:
    """Stops training early if validation loss doesn't improve."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience, self.stop_epoch, self.verbose = patience, stop_epoch, verbose
        self.counter, self.best_score, self.early_stop = 0, None, False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss improved ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args, writer=None):
    """Train a single fold."""
    print(f'🚀 Training Fold {cur}...')
    start_time = time.time()
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)
    
    if args.log_data:
        writer = SummaryWriter(writer_dir, flush_secs=10)
    
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, f'splits_{cur}.csv'))
    
    print(f'Training on {len(train_split)} samples | Validating on {len(val_split)} samples | Testing on {len(test_split)} samples')

    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    model = CLAM_SB(**model_dict) if args.model_type == 'clam_sb' else CLAM_MB(**model_dict)
    model.to(device)  # 🔥 Move model to GPU
    optimizer = get_optim(model, args)
    
    train_loader, val_loader, test_loader = get_split_loader(train_split, training=True), get_split_loader(val_split), get_split_loader(test_split)
    
    early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True) if args.early_stopping else None

    # Initialize log storage for training and validation metrics
    logs = {
        'epoch': [],
        'train_loss': [],
        'train_inst_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_error': [],
        'val_inst_loss': [],
    }

    
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        train_loss, train_inst_loss = train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
        val_result = validate_clam(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)
        
        if isinstance(val_result, bool):
            stop = val_result  # early stopping triggered
            break
        else:
            val_loss, val_auc, val_error, val_inst_loss = val_result
            stop = False

        # Store metrics per epoch
        logs['epoch'].append(epoch)
        logs['train_loss'].append(train_loss)
        logs['train_inst_loss'].append(train_inst_loss)
        logs['val_loss'].append(val_loss)
        logs['val_auc'].append(val_auc)
        logs['val_error'].append(val_error)
        logs['val_inst_loss'].append(val_inst_loss)

        print(f'🕒 Epoch {epoch} completed in {time.time() - epoch_start:.2f} seconds')
        if stop: break
    
    model.load_state_dict(torch.load(os.path.join(args.results_dir, f's_{cur}_checkpoint.pt')))
    # results_dict, test_auc, val_auc, acc_logger = summary(model, test_loader, args.n_classes)
    # i change this 
    results_dict, test_auc, val_auc, acc_logger = summary(model, test_loader, args.n_classes, cur, args.results_dir)

    print(f'✅ Training Completed for Fold {cur} | Total Time: {time.time() - start_time:.2f} seconds')
    # return results_dict, test_auc, val_auc, acc_logger
    
    test_acc = acc_logger.get_summary(0)[0]  # Accuracy for class 0
    val_acc = acc_logger.get_summary(1)[0]   # Accuracy for class 1

    print(f"Returning from train(): test_auc={test_auc}, val_auc={val_auc}, test_acc={test_acc}, val_acc={val_acc}")

    log_df = pd.DataFrame(logs)
    log_df.to_csv(os.path.join(writer_dir, 'training_log.csv'), index=False)
    
    return results_dict, test_auc, val_auc, test_acc, val_acc  # ✅ Now returns 5 values
   

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss, train_error, train_inst_loss, inst_count = 0., 0., 0., 0
    
    print(f'🚀 Starting Epoch {epoch}...')
    
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, _, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        loss = loss_fn(logits, label)
        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss
        train_loss += loss.item()
        train_inst_loss += instance_loss.item()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 50 == 0:
            print(f'[Epoch {epoch} | Batch {batch_idx+1}] Loss: {loss.item():.4f}, Instance Loss: {instance_loss.item():.4f}, Total Loss: {total_loss.item():.4f}')
    
    train_loss /= len(loader)
    train_inst_loss /= inst_count
    print(f'✅ Epoch {epoch} Completed | Train Loss: {train_loss:.4f} | Train Clustering Loss: {train_inst_loss:.4f}')
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        
    return train_loss, train_inst_loss



   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            inst_count += 1
            val_inst_loss += instance_loss.item()

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if inst_count > 0:
        val_inst_loss /= inst_count

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))

    print(f"\n✅ Epoch {epoch} Completed | Val Loss: {val_loss:.4f} | Val Clustering Loss: {val_inst_loss:.4f} | Val Error: {val_error:.4f}")

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        
        if writer and acc is not None:
            writer.add_scalar(f'val/class_{i}_acc', acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("🚨 Early stopping triggered!")
            return True

    return val_loss, auc, val_error, val_inst_loss


# def summary(model, loader, n_classes):
def summary(model, loader, n_classes, fold_id, results_dir):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({
            slide_id: {
                'slide_id': np.array(slide_id),
                'prob': probs,
                'label': label.item()
            }
        })

        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))

    # 🔽 Save predictions in the correct fold directory
    results_df = pd.DataFrame({
        'slide_id': slide_ids,
        'Y': all_labels,
        'Y_hat': all_preds
    })

    # Detect and use correct fold subdir (e.g., /.../tumor_vs_normal_CLAM_s1/0/)
    # This will be passed indirectly via the model checkpoint path
    # You can update this if needed to use args.results_dir/cur directly
    if hasattr(loader.dataset, 'fold_dir'):
        fold_dir = loader.dataset.fold_dir  # optional if you pass it in the dataset
    else:
        # fallback to saving inside the parent fold directory based on training script logic
        fold_path = os.path.join(results_dir, str(fold_id))
        os.makedirs(fold_path, exist_ok=True)


        fold_dir = fold_path

    results_df.to_csv(os.path.join(fold_path, "predictions.csv"), index=False)
    print(f"✅ Saved predictions: {os.path.join(fold_path, 'predictions.csv')}")

    return patient_results, test_error, auc, acc_logger

# def summary(model, loader, n_classes):
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     model.eval()
#     test_loss = 0.
#     test_error = 0.

#     all_probs = np.zeros((len(loader), n_classes))
#     all_labels = np.zeros(len(loader))

#     slide_ids = loader.dataset.slide_data['slide_id']
#     patient_results = {}

#     for batch_idx, (data, label) in enumerate(loader):
#         data, label = data.to(device), label.to(device)
#         slide_id = slide_ids.iloc[batch_idx]
#         with torch.inference_mode():
#             logits, Y_prob, Y_hat, _, _ = model(data)

#         acc_logger.log(Y_hat, label)
#         probs = Y_prob.cpu().numpy()
#         all_probs[batch_idx] = probs
#         all_labels[batch_idx] = label.item()
        
#         patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
#         error = calculate_error(Y_hat, label)
#         test_error += error

#     test_error /= len(loader)

#     if n_classes == 2:
#         auc = roc_auc_score(all_labels, all_probs[:, 1])
#         aucs = []
#     else:
#         aucs = []
#         binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
#         for class_idx in range(n_classes):
#             if class_idx in all_labels:
#                 fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
#                 aucs.append(calc_auc(fpr, tpr))
#             else:
#                 aucs.append(float('nan'))

#         auc = np.nanmean(np.array(aucs))


#     return patient_results, test_error, auc, acc_logger
