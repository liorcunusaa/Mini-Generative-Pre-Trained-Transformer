# Build a Mini GPT Model

Proyek ini adalah implementasi model **Generative Pre-trained Transformer (GPT)** yang sederhana dan edukatif, dibangun dari nol menggunakan PyTorch. Model ini dirancang untuk tujuan pembelajaran dan memahami arsitektur dasar transformer.

## Daftar Isi

- [Ringkasan Proyek](#ringkasan-proyek)
- [Struktur Direktori](#struktur-direktori)
- [Komponen Utama](#komponen-utama)
- [Persyaratan](#persyaratan)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Penjelasan Algoritma](#penjelasan-algoritma)
- [Hasil dan Generasi Teks](#hasil-dan-generasi-teks)
- [Kontribusi](#kontribusi)

## Ringkasan Proyek

Model TinyGPT adalah implementasi transformer yang dioptimalkan untuk pemahaman konsep dasar GPT. Model ini menggunakan:

- **Self-Attention Mechanism**: Untuk memahami hubungan antar kata dalam teks
- **Multi-Head Attention**: Untuk menangkap berbagai representasi hubungan antar kata
- **Feed Forward Network**: Untuk pemrosesan non-linear
- **Positional Encoding**: Untuk mempertahankan informasi posisi token dalam sequence

Model dilatih pada corpus kalimat sederhana dan dapat menghasilkan teks baru berdasarkan input awal.

## Struktur Direktori

```
Build-a-Mini-GPT-Model/
├── module_0/
│   ├── l-1_language_models/
│   │   └── [Modul model bahasa dasar]
│   └── l-2_tinyGPT/
│       ├── demo.py                 # Script utama untuk melatih dan menjalankan model
│       ├── transformer_blocks.py   # Definisi blok-blok transformer
│       └── __pycache__/           # Cache Python (bisa diabaikan)
└── README.md                       # Dokumentasi proyek (file ini)
```

## Komponen Utama

### 1. **SelfAttentionHead**

Implementasi single head dari self-attention mechanism:

```python
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size)
    def forward(self, x)
```

**Cara kerja**:

- Mengubah input menjadi Query (Q), Key (K), dan Value (V)
- Menghitung attention weights: `wei = Q @ K^T / sqrt(d_k)`
- Menerapkan masking untuk mencegah model melihat token di masa depan
- Menghitung output: `output = wei @ V`

### 2. **MultiHeadAttention**

Menggabungkan multiple attention heads untuk menangkap berbagai representasi:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, num_heads)
    def forward(self, x)
```

**Cara kerja**:

- Menjalankan `num_heads` dari SelfAttentionHead secara paralel
- Menggabungkan output dari semua heads (concatenation)
- Memproyeksikan hasil ke embedding dimension asli

### 3. **FeedForward Network**

Jaringan feed-forward untuk pemrosesan non-linear:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd)
    def forward(self, x)
```

**Arsitektur**:

- Linear layer: `n_embd → 4*n_embd`
- Aktivasi ReLU untuk non-linearitas
- Linear layer: `4*n_embd → n_embd`

### 4. **Transformer Block**

Menggabungkan attention dan feed-forward dengan residual connections dan layer normalization:

```python
class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads)
    def forward(self, x)
```

**Alur**:

```
Input → LayerNorm → MultiHeadAttention → Residual Add
   → LayerNorm → FeedForward → Residual Add → Output
```

### 5. **TinyGPT Model**

Model utama yang mengkombinasikan semua komponen:

```python
class TinyGPT(nn.Module):
    def __init__(self)
    def forward(self, idx, targets=None)
    def generate(self, idx, max_new_tokens)
```

**Komponen**:

- `token_embedding`: Mengubah token ID menjadi embedding vectors
- `position_embedding`: Menambahkan informasi posisi relatif
- `blocks`: Stack dari `n_layers` transformer blocks
- `ln_f`: Final layer normalization
- `head`: Linear layer untuk prediksi token selanjutnya

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Build-a-Mini-GPT-Model.git
cd Build-a-Mini-GPT-Model
```

### 2. Install Dependencies

```bash
pip install torch numpy
```

Atau jika menggunakan GPU CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 💻 Penggunaan

### 1. Jalankan Demo Script

```bash
cd module_0/l-2_tinyGPT
python demo.py
```

### 2. Kustomisasi Hyperparameter

Edit `demo.py` untuk mengubah hyperparameter:

```python
block_size = 6              # Panjang sequence input (context length)
embedding_dim = 32          # Dimensi embedding
n_heads = 2                 # Jumlah attention heads
n_layers = 2                # Jumlah transformer blocks
lr = 1e-3                   # Learning rate
epochs = 1500               # Jumlah epoch training
```

### 3. Modifikasi Corpus

Ubah `corpus` di `demo.py` untuk melatih dengan data berbeda:

```python
corpus = [
    "hello everyone, good morning",
    "my name is lisa",
    # Tambahkan kalimat Anda di sini
]
```

## 🧠 Penjelasan Algoritma

### Self-Attention Mechanism

Self-attention memungkinkan setiap token dalam sequence untuk berkomunikasi dengan semua token lainnya. Formula dasarnya:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Dimana:

- **Q (Query)**: Representasi dari kata yang sedang diproses
- **K (Key)**: Representasi dari semua kata untuk matching
- **V (Value)**: Informasi aktual yang akan dikombinasikan
- **$\sqrt{d_k}$**: Faktor normalisasi untuk stabilitas numerik

### Multi-Head Attention

Menjalankan attention mechanism secara paralel dengan sub-representasi yang berbeda:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Dimana setiap head menggunakan subset dari embedding dimensions.

### Positional Encoding

Menambahkan informasi posisi ke token embeddings sehingga model dapat memahami urutan:

```
position_encoding = embedding(position) untuk setiap posisi
x = token_embedding + position_embedding
```

### Training Loop

Model dilatih menggunakan **cross-entropy loss** dengan optimasi menggunakan **AdamW**:

1. Ambil batch dari data: `get_batch()`
2. Forward pass: `logits, loss = model(xb, yb)`
3. Backward pass: `loss.backward()`
4. Update weights: `optimizer.step()`

## 📊 Hasil dan Generasi Teks

Setelah training, model dapat menghasilkan teks baru:

```python
context = torch.tensor([[word2idx["hello"]]], dtype=torch.long)
out = model.generate(context, max_new_tokens=15)
print(" ".join(idx2word[int(i)] for i in out[0]))
```

**Contoh Output**:

```
Generated text:
hello everyone good morning the roads of Unesa are busy the train is late
```

### Proses Generasi

1. Mulai dengan token awal (context)
2. Jalankan model untuk mendapatkan prediksi token selanjutnya
3. Sample dari distribusi probabilitas menggunakan `multinomial sampling`
4. Tambahkan token baru ke sequence
5. Ulangi sampai panjang maksimal tercapai

## 🔍 Penjelasan Parameter

| Parameter       | Deskripsi                                        | Default |
| --------------- | ------------------------------------------------ | ------- |
| `block_size`    | Panjang context untuk prediksi (sequence length) | 6       |
| `embedding_dim` | Dimensi embedding vectors                        | 32      |
| `n_heads`       | Jumlah attention heads                           | 2       |
| `n_layers`      | Jumlah transformer blocks                        | 2       |
| `lr`            | Learning rate untuk optimizer                    | 1e-3    |
| `epochs`        | Jumlah iterasi training                          | 1500    |
| `batch_size`    | Ukuran batch dalam `get_batch()`                 | 16      |

## 📈 Improvement dan Saran

Untuk meningkatkan performa model:

1. **Tambah Data**: Gunakan corpus yang lebih besar dan bervariasi
2. **Tune Hyperparameter**: Eksperimen dengan embedding_dim, n_heads, dan n_layers
3. **Validation Set**: Buat validation loop untuk monitor overfitting
4. **Checkpointing**: Simpan model checkpoint selama training
5. **Learning Rate Scheduling**: Gunakan learning rate scheduler untuk convergence lebih baik
6. **Dropout**: Tambahkan dropout layers untuk regularisasi
7. **Beam Search**: Implementasi beam search untuk generasi teks yang lebih baik

## 🐛 Debugging

### Model Tidak Konvergen

- Cek learning rate, coba kurangi atau naikkan
- Pastikan data sudah di-normalize dengan benar
- Cek gradient flow dengan gradient clipping

### GPU Memory Issues

- Kurangi batch_size
- Kurangi embedding_dim atau n_layers
- Gunakan gradient accumulation

### Poor Generation Quality

- Tingkatkan corpus size
- Tambah epochs
- Tingkatkan model capacity (embedding_dim, n_layers)

## 📚 Referensi

- **Attention Is All You Need**: Vaswani et al., 2017
  - Paper: https://arxiv.org/abs/1706.03762
- **Language Models are Unsupervised Multitask Learners**: Radford et al., 2019
  - Paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html

## 📄 Lisensi

Proyek ini tersedia di bawah lisensi MIT. Silakan gunakan dan modifikasi untuk tujuan pembelajaran maupun komersial.

## 👥 Kontribusi

Kontribusi sangat diterima! Berikut cara berkontribusi:

1. Fork repository
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

## ❓ FAQ

**Q: Bagaimana cara menggunakan model yang sudah dilatih?**
A: Simpan model menggunakan `torch.save(model.state_dict(), 'model.pth')` dan load dengan `model.load_state_dict(torch.load('model.pth'))`.

**Q: Bisakah saya menggunakan dataset yang lebih besar?**
A: Ya, ubah `corpus` dengan data yang lebih besar. Untuk dataset sangat besar, pertimbangkan streaming data atau chunking.

**Q: Bagaimana performa pada GPU vs CPU?**
A: GPU akan jauh lebih cepat untuk model yang lebih besar. Pindahkan model ke GPU dengan `.to('cuda')`.

**Q: Apakah model dapat melakukan fine-tuning?**
A: Ya, Anda dapat melanjutkan training dari checkpoint yang tersimpan atau transfer learning.

---

**Dibuat dengan ❤️ untuk pembelajaran AI dan Deep Learning**

Terakhir diupdate: Januari 2026
