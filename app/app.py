import os
import torch
import torch.nn as nn
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_dataloaders, train_transform, val_transform, test_transform
from src.model import BTCCnn
from src.train import train, evaluate
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.predict import predict_samples
from app.utils import plot_training_curves



# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA = os.path.join(BASE_DIR, "dataset", "preprocessed", "Training")
TEST_DATA = os.path.join(BASE_DIR, "dataset", "preprocessed", "Testing")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Device: {device}")

# --- Data ---
train_loader, val_loader, test_loader = get_dataloaders(
    TRAIN_DATA, TEST_DATA,
    train_transform, val_transform, test_transform
)

# --- Model ---
model = BTCCnn(num_classes=4, in_ch=1).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# --- Train ---
t_losses, t_accs, v_losses, v_accs = train(
    model, train_loader, val_loader,
    device, optimizer, criterion, scheduler, epochs=30
)
plot_training_curves(t_losses, t_accs, v_losses, v_accs)

# --- Load Best Model ---
model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "models", "checkpoints", "best_model.pth"),
    map_location=device, weights_only=True
))
model.to(device)

# --- Evaluate ---
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device)
plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES)
plot_roc_curve(all_labels, all_probs, CLASS_NAMES)

# --- Predict Samples ---
predict_samples(TEST_DATA, test_transform, device, model, CLASS_NAMES)