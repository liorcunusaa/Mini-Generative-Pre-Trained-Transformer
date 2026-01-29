import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Import block transformer digunakan untuk membangun model TinyGPT
from transformer_blocks import Block

# Setup device dan paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Device:", device)

corpus = [
    "hello everyone, good morning",
    "my name is lisa",
    "the tea is very hot",
    "the roads of Unesa are busy",
    "it is raining in",
    "the train is late again",
    "i love a rainy day",
    "i love eating seblak and drinking matcha",
    "Persebaya is a football club from Surabaya"
]

corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
print (text)

words = list(set(text.split()))
print(words)

vocab_size = len(words)
print(vocab_size)

word2idx = {w: i for i, w in enumerate(words)}
print("word2idx : ", word2idx)

idx2word = {i: w for w, i in word2idx.items()} 
print("idx2word : ", idx2word) 

data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
print("data : ", data) 
print(len(data))

# ===== HYPERPARAMETERS =====
block_size = 6 
embedding_dim = 32
n_heads = 2  
n_layers = 2 
lr = 1e-3 
epochs = 200  # SHORTER FOR TESTING
batch_size = 16
eval_interval = 20  # SHORTER INTERVAL

# ===== TRAIN/VAL SPLIT =====
split_idx = int(0.8 * len(data))  # 80% train, 20% val
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"\nData split: train={len(train_data)}, val={len(val_data)}")

def get_batch(data_split='train', batch_size=16):
    """Ambil batch dari train atau val set"""
    current_data = train_data if data_split == 'train' else val_data
    ix = torch.randint(len(current_data) - block_size, (batch_size,))  
    x = torch.stack([current_data[i:i+block_size] for i in ix])  
    y = torch.stack([current_data[i+1:i+block_size+1] for i in ix]) 
    return x.to(device), y.to(device)

def compute_loss(data_split='val', num_batches=5):
    """Compute average loss pada dataset"""
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            xb, yb = get_batch(data_split, batch_size)
            _, loss = model(xb, yb)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
    
    def save_model(self, filepath):
        """Simpan model ke file"""
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model dari file"""
        self.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Model loaded from {filepath}")

# ===== INITIALIZE MODEL =====
model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Lists untuk tracking loss
train_losses = []
val_losses = []
steps = []

print("\n" + "="*50)
print("STARTING TRAINING (QUICK TEST)")
print("="*50)

for step in range(epochs):
    # Training step
    model.train()
    xb, yb = get_batch('train', batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # Evaluation step
    if step % eval_interval == 0:
        val_loss = compute_loss('val', num_batches=3)
        val_losses.append(val_loss)
        steps.append(step)
        
        print(f"Step {step:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

# ===== SAVE MODEL & VOCAB =====
print("\n" + "="*50)
print("TRAINING COMPLETED")
print("="*50)

model_path = model_dir / 'tinygpt_model.pth'
model.save_model(str(model_path))

# Simpan vocabulary
import json
vocab_path = model_dir / 'vocab.json'
vocab_data = {
    'word2idx': word2idx,
    'idx2word': {str(k): v for k, v in idx2word.items()}
}
with open(vocab_path, 'w') as f:
    json.dump(vocab_data, f, indent=2)
print(f"Vocabulary saved to {vocab_path}")

# ===== PLOT TRAINING CURVES =====
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

plt.figure(figsize=(12, 6))
plt.plot(range(len(train_losses)), train_losses, label='Train Loss', alpha=0.7)
plt.plot(steps, val_losses, label='Validation Loss', marker='o', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('TinyGPT Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plot_path = model_dir / 'training_curve.png'
plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
print(f"Training curve saved to {plot_path}")
plt.close()

print(f"\n Files created:")
print(f"   - Model: {model_path}")
print(f"   - Vocab: {vocab_path}")
print(f"   - Plot: {plot_path}")

# ===== INFERENCE =====
print("\n" + "="*50)
print("GENERATING TEXT")
print("="*50)

context = torch.tensor([[word2idx["hello"]]], dtype=torch.long).to(device)
out = model.generate(context, max_new_tokens=15)

print("\nGenerated text:")
print(" ".join(idx2word[int(i)] for i in out[0]))

print("\n QUICK TEST COMPLETED SUCCESSFULLY!")
