import os
import torch

class EarlyStopping():
    def __init__(self, patience= 7, min_delta= 0):
        self.patience= patience
        self.min_delta = min_delta
        self.counter= 0
        self.best_score= float("inf")
        self.early_stop= False

    def __call__(self, val_loss, model):
        if val_loss < self.best_score - self.min_delta:
            self.best_score= val_loss
            self.counter = 0
            os.makedirs("../models/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "../models/checkpoints/best_model.pth")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop= True

def train_one_epoch(model,
                    data_loader,
                    device,
                    optimizer,
                    criterion):
    model.train()

    train_loss= 0
    correct, total= 0, 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs= model(images)
        loss= criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()*labels.size(0)
        predicted = torch.argmax(outputs, dim= 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss/total
    train_acc = 100* correct/total
    return avg_loss, train_acc

def evaluate(model,
             data_loader,
             device,
             criterion ):
    model.eval()

    val_loss= 0
    correct, total= 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels= images.to(device), labels.to(device)

            outputs= model(images)
            loss= criterion(outputs, labels)

            val_loss += loss.item()*labels.size(0)
            predicted = torch.argmax(outputs, dim= 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = val_loss/total
    val_acc = 100*correct/total

    return avg_loss, val_acc


def train(model,
          train_loader,
          val_loader,
          device,
          optimizer,
          criterion,
          scheduler,
          epochs=30):

    early_stopping = EarlyStopping(patience=7, min_delta=0.001)

    t_losses, t_accs = [], []
    v_losses, v_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        early_stopping(val_loss, model)

        t_losses.append(train_loss)
        t_accs.append(train_acc)
        v_losses.append(val_loss)
        v_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} -> Train Loss: {train_loss:.2f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.2f}, Acc: {val_acc:.2f}%")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return t_losses, t_accs, v_losses, v_accs








