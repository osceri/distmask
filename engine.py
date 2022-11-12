import torch
from torch.nn import functional as F
from sklearn import metrics
import numpy as np

def train_one_epoch(model : torch.nn.Module, 
        data_loader, optimizer : torch.optim.Optimizer, 
        device : torch.device, epoch : int, 
        loss_scaler = None, args = None, logger = lambda x: None):

    train_loss = 0

    model.train()

    optimizer.zero_grad()

    count = 0
    length = len(data_loader)

    for (samples, target) in data_loader:
        targets_mask = target[:, 20:]
        targets = target[:, :20]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_mask = targets_mask.to(device, non_blocking=True)
        targets = targets > 0.5
        targets = targets.float()

        outputs = model(samples)
        samples_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
        samples_loss = targets_mask * samples_loss
        loss = samples_loss.mean()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        train_loss += loss_value

        if count % (length // 3) == 0:
            logger(f'Epoch {epoch} : {count}/{length}')

        count = count + 1

    logger(f'Epoch {epoch} : {length}/{length}')

    return train_loss

def train_one_epoch_student(model : torch.nn.Module, 
        data_loader, optimizer : torch.optim.Optimizer, 
        device : torch.device, epoch : int, 
        loss_scaler = None, args = None, logger = lambda x: None):
    return None

def evaluate(model : torch.nn.Module, 
        data_loader, optimizer : torch.optim.Optimizer, 
        device : torch.device, epoch : int, 
        loss_scaler = None, args = None, logger = lambda x: None):

    eval_loss = 0

    model.eval()

    alloutput = []
    alltarget = []
    allmask = []

    count = 0
    length = len(data_loader)

    for (samples, target) in data_loader:
        targets_mask = target[:, 20:]
        targets = target[:, :20]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_mask = targets_mask.to(device, non_blocking=True)
        targets = targets > 0.5
        targets = targets.float()

        outputs = model(samples)
        samples_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
        samples_loss = targets_mask * samples_loss
        loss = samples_loss.mean()
        loss.backward()

        outputs = torch.sigmoid(outputs).cpu().detach()
        targets = targets.cpu().detach()
        targets_mask = targets_mask.cpu().detach()

        loss_value = loss.item()
        eval_loss += loss_value

        alloutput.append(outputs)
        alltarget.append(targets)
        allmask.append(targets_mask)

        if count % (length // 3) == 0:
            logger(f'Evaluate {epoch} : {count}/{length}')

        count = count + 1


    alloutput = np.concatenate(alloutput, 0)
    alltarget = np.concatenate(alltarget, 0)
    allmask = np.concatenate(allmask, 0)

    average_precision = np.array([
        metrics.average_precision_score(
            alltarget[:, i], alloutput[:, i], sample_weight=allmask[:, i]
        ) for i in range(alloutput.shape[1])
    ])
    mAP = average_precision.mean().item()

    #error = np.linalg.norm(alloutput.ravel() - alltarget.ravel())
    #print(error)

    return eval_loss, mAP

