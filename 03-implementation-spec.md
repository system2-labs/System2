# System2 — Implementation Specification

**Status:** Implementation plan v0.1
**Created:** 2026-09-24

---

## 1. Architecture

### 1.1 Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  User Query (text)                                                   │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TOKENIZER (Qwen3Tokenizer or MistralTokenizer)                     │
│  Text → Token IDs                                                  │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENCODER (ModernBERT-large, 395M params)                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Token embeddings → LayerNorm → Multi-head attention layers │   │
│  │ → SwiGLU activations → RoPE positional encodings           │   │
│  │ → Last-token embedding (768-dim hidden state)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────────────────────────┘
               │ (hidden state vector)
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  N PARALLEL THOUGHT HEADS (5-10, ~40M each)                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Head 1: ──────────────────────────────────────────────────┐  │  │
│  │          Input → Linear(768→512) → LayerNorm              │  │  │
│  │          → Attention(64 heads, 512 dim) → LayerNorm       │  │  │
│  │          → FeedForward(512→2048→512, SwiGLU) → LayerNorm  │  │  │
│  │          → Linear(512→vocab_size)                          │  │  │
│  │          → Softmax → Token probabilities                   │  │  │
│  │          → Sample top-k tokens → "Hypothesis A" + conf    │  │  │
│  └─────────────────────────────────────────────────────────────┘  │  │
│  │ Head 2: (identical architecture, different weights)             │  │  │
│  │ Head 3: (identical architecture, different weights)             │  │  │
│  │ ...                                                             │  │  │
│  │ Head N: (identical architecture, different weights)             │  │  │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────────────────────────┘
               │ (hypothesis_1, conf_1), (hypothesis_2, conf_2), ...
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  JUDGER/LAAYER (~50M params)                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Input: [encoder_hidden, hypothesis_1_emb, hypothesis_2_emb │   │   │
│  │        ..., hypothesis_N_emb]                               │   │
│  │ → Linear → LayerNorm → Cross-Attention (encoder→heads)     │   │
│  │ → Self-Attention on all hypotheses                         │   │
│  │ → Output: (selected_hypothesis, calibrated_confidence)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT: (hypothesis, confidence_score, pruned_count)               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Decisions

1. **No autoregressive token generation.** The model doesn't generate text token-by-token. Each thought head produces a *single hypothesis* in one forward pass.

2. **Calibrated confidence.** Each hypothesis comes with a confidence score trained via RLCD (strict proper scoring rules). Honest probability reporting maximizes expected reward.

3. **Pruning, not averaging.** Weak hypotheses are discarded (below a confidence threshold), not averaged into a consensus. This is deliberate — averaging dilutes reasoning quality.

4. **The judger selects, not generates.** The judger doesn't produce new text. It scores existing hypotheses and selects the best one.

---

## 2. Component Specifications

### 2.1 Tokenizer

**Option A: Qwen3Tokenizer (recommended)**
- Tokenizer used by Qwen3.8 models
- Vocabulary: ~151,000 tokens
- Max sequence: 32K tokens
- Pros: Well-tested, handles English + code well
- Cons: Tied to Qwen ecosystem

**Option B: MistralTokenizer**
- Tokenizer used by Mistral models
- Vocabulary: ~32,000 tokens
- Max sequence: 32K tokens
- Pros: Simpler, widely used
- Cons: Smaller vocabulary, may use more tokens for same text

**Option C: Custom (long-term)**
- Train a tokenizer on your own data
- Vocabulary size: 32K-50K (balanced between compression and expressiveness)
- Pros: Optimized for your domain
- Cons: Requires training data, adds complexity

**Recommendation:** Start with Qwen3Tokenizer (Option A). You can always swap later.

### 2.2 Encoder (ModernBERT-large)

```python
from transformers import AutoModel

class Encoder(nn.Module):
    """ModernBERT-large encoder, frozen after pre-training.
    
    Architecture:
    - Input: token IDs (up to 1024 tokens)
    - Output: last-token hidden state (768-dim)
    - Parameters: 395M
    - 4-bit size: ~1.6GB
    
    Reference: ModernBERT (Shoeybi et al., 2024)
    """
    
    def __init__(self, model_name="answerai/ModernBERT-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        """Returns last-token hidden state (pooled output)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Pool using [CLS] token (or last token, depending on BERT variant)
        return outputs.pooler_output  # (batch_size, 768)
```

**Implementation notes:**
- Load with `4-bit` quantization (GPTQ or bitsandbytes NF4)
- Freeze all encoder weights after pre-training (Phase 1 complete)
- No gradient computation through encoder during head/judger training
- Input: token IDs, output: 768-dim vector

### 2.3 Thought Heads

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class ThoughtHead(nn.Module):
    """Single thought head: processes encoder output and produces
    a hypothesis with calibrated confidence.
    
    Architecture:
    - Input: encoder's last-token hidden state (768-dim)
    - Internal: 2 transformer layers (512 dim, 64 heads)
    - Output: (hypothesis_text, confidence_score)
    - Parameters: ~40M per head
    
    Design rationale:
    - Small transformer (2 layers) to keep parameter count low
    - Multiple attention heads (64) for expressive capacity
    - SwiGLU activations (ModernBERT optimization)
    - Calibrated via RLCD training
    """
    
    def __init__(self, hidden_size=768, head_dim=512, num_attention_heads=64, 
                 num_hidden_layers=2, vocab_size=151000):
        super().__init__()
        
        self.input_proj = nn.Linear(hidden_size, head_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=head_dim,
                nhead=num_attention_heads,
                dim_feedforward=head_dim * 4,  # 2048
                activation='swiglu',  # ModernBERT optimization
                batch_first=True
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Confidence scoring head (like Laya's option-marker scorer)
        self.confidence_head = nn.Sequential(
            nn.Linear(head_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),  # Single scalar: log-odds of confidence
        )
        
        # Hypothesis generation head (maps to vocabulary)
        self.hypothesis_head = nn.Sequential(
            nn.Linear(head_dim, vocab_size)
        )
        
    def forward(self, encoder_output):
        """
        Args:
            encoder_output: (batch_size, 768) — last-token hidden state
            
        Returns:
            (hypothesis_token, confidence_score)
            - hypothesis_token: int — selected hypothesis token index
            - confidence_score: float — calibrated confidence (0-1)
        """
        # Project to head dimension
        x = self.input_proj(encoder_output)  # (batch_size, 512)
        
        # Transformer layers (2 layers, 64 attention heads)
        x = x.unsqueeze(1)  # (batch_size, 1, 512) — single token
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(1)  # (batch_size, 512)
        
        # Confidence scoring
        logit = self.confidence_head(x).squeeze(-1)  # (batch_size,)
        confidence = torch.sigmoid(logit)  # (batch_size,) — probability
        
        # Hypothesis generation (top-5 candidates for selection)
        logits = self.hypothesis_head(x)  # (batch_size, vocab_size)
        hypothesis_token = logits.argmax(dim=-1).item()  # int
        
        return hypothesis_token, confidence.item()
```

**Implementation notes:**
- Each head has identical architecture but different weights
- Initialize with Xavier uniform (random weights)
- During inference, top-5 token candidates are returned (not just argmax)
- Confidence is calibrated via RLCD: if confidence = 0.85, the hypothesis is correct ~85% of the time
- 4-bit quantization: ~40M parameters → ~160MB per head
- 10 heads: ~1.6GB total

### 2.4 Judger Layer

```python
class Judger(nn.Module):
    """Judger/pruner: evaluates all hypotheses and selects the best one.
    
    Architecture:
    - Input: encoder hidden state + all hypothesis embeddings
    - Cross-attention: encoder queries all hypotheses
    - Self-attention: hypotheses attend to each other (for comparison)
    - Output: (selected_hypothesis, calibrated_confidence, pruned_count)
    - Parameters: ~50M
    
    Design rationale:
    - Cross-attention allows encoder to compare hypotheses
    - Self-attention allows hypotheses to "compete" with each other
    - Pruning threshold: hypotheses below confidence drop
    """
    
    def __init__(self, hidden_size=768, head_dim=512, num_heads=32,
                 num_layers=2, num_hypotheses=10):
        super().__init__()
        
        self.encoder_proj = nn.Linear(hidden_size, head_dim)
        self.hypothesis_proj = nn.Linear(head_dim, head_dim)  # from heads
        
        # Cross-attention: encoder queries hypotheses
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=head_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Self-attention: hypotheses compare with each other
        self.self_attention = nn.MultiheadAttention(
            embed_dim=head_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output head: confidence calibration (RLCD trained)
        self.output_head = nn.Sequential(
            nn.Linear(head_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),  # log-odds of confidence
        )
        
        self.num_hypotheses = num_hypotheses
        
    def forward(self, encoder_hidden, hypothesis_embeddings, 
                attention_mask=None):
        """
        Args:
            encoder_hidden: (batch_size, 768) — encoder output
            hypothesis_embeddings: (batch_size, num_hypotheses, head_dim)
            attention_mask: (batch_size, num_hypotheses) — which hypotheses are valid
            
        Returns:
            (selected_hypothesis_idx, confidence_score, pruned_count)
        """
        # Project encoder to head dimension
        encoder_proj = self.encoder_proj(encoder_hidden).unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention: encoder queries all hypotheses
        attended, _ = self.cross_attention(
            encoder_proj,  # query
            hypothesis_embeddings.transpose(0, 1),  # key, value (D, B, num_hyps)
            key=hypothesis_embeddings.transpose(0, 1),
            value=hypothesis_embeddings.transpose(0, 1)
        )
        attended = attended.squeeze(1)  # (B, D)
        
        # Self-attention: hypotheses compete with each other
        # (transposed for multi-head attention)
        hyp_attended, _ = self.self_attention(
            hypothesis_embeddings.transpose(0, 1),  # (D, B, num_hyps)
            hypothesis_embeddings.transpose(0, 1),  # key
            hypothesis_embeddings.transpose(0, 1),  # value
            key_padding_mask=~attention_mask  # mask invalid hypotheses
        )
        hyp_attended = hyp_attended.transpose(0, 1)  # (B, num_hyps, D)
        
        # Fuse: concatenate encoder's attended view + all hypothesis embeddings
        fused = torch.cat([attended.unsqueeze(1), hyp_attended], dim=1)  # (B, 1+num_hyps, D)
        
        # Select best hypothesis
        scores = fused[:, 1:].mean(dim=-1)  # (B, num_hyps) — average embedding similarity
        selected_idx = scores.argmax(dim=-1).item()  # int
        
        # Calibrated confidence (via RLCD training)
        selected_emb = hyp_attended[:, selected_idx]  # (B, D)
        logit = self.output_head(selected_emb).squeeze(-1)  # (B,)
        confidence = torch.sigmoid(logit).item()  # (0-1)
        
        # Count pruned hypotheses (below confidence threshold)
        threshold = 0.3  # configurable
        pruned_count = (scores < confidence - 0.3).sum().item()  # approximate
        
        return selected_idx, confidence, pruned_count
```

**Implementation notes:**
- 50M parameters (4-bit: ~200MB)
- Cross-attention lets encoder compare hypotheses directly
- Self-attention lets hypotheses "compete" with each other
- Confidence threshold: configurable (default 0.3)
- Pruned hypotheses are excluded from final output

### 2.5 System2 Class (Full Model)

```python
class System2(nn.Module):
    """System2: Encoder + parallel thought heads + judger.
    
    Architecture:
    - Encoder: ModernBERT-large (frozen, 395M params)
    - Thought heads: 5-10 parallel transformers (~40M each)
    - Judger: cross-attention + self-attention (~50M)
    - Total: ~500M parameters (4-bit: ~2GB)
    
    Inference:
    - Single forward pass through encoder (768-dim output)
    - All thought heads process encoder output in parallel
    - Each head produces (hypothesis, confidence) pair
    - Judger selects best hypothesis, prunes weak ones
    - Returns: (hypothesis, confidence, pruned_count)
    
    Latency: ~100-500ms (depending on number of heads)
    RAM: ~2GB (4-bit quantization)
    """
    
    def __init__(self, num_heads=10, vocab_size=151000):
        super().__init__()
        
        # Encoder (frozen after pre-training)
        self.encoder = Encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Thought heads
        self.heads = nn.ModuleList([
            ThoughtHead(vocab_size=vocab_size) 
            for _ in range(num_heads)
        ])
        
        # Judger
        self.judger = Judger(num_hypotheses=num_heads)
        
        self.num_heads = num_heads
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len) — token IDs
            attention_mask: (batch_size, seq_len) — attention mask
            
        Returns:
            (hypothesis_text, confidence_score, pruned_count, all_hypotheses)
            - hypothesis_text: str — selected hypothesis
            - confidence_score: float — calibrated confidence (0-1)
            - pruned_count: int — number of hypotheses pruned
            - all_hypotheses: list of (hypothesis, confidence) — all hypotheses
        """
        # Step 1: Encode
        encoder_output = self.encoder(input_ids, attention_mask)  # (B, 768)
        
        # Step 2: All thought heads process in parallel
        hypotheses = []
        confidences = []
        hypothesis_embeddings = []  # for judger
        
        for head in self.heads:
            h_token, h_confidence = head(encoder_output)
            hypotheses.append(h_token)
            confidences.append(h_confidence)
            
            # Get hypothesis embedding from head's output
            # (we need to re-run the head to get the embedding)
            h_emb = self._get_hypothesis_embedding(head, encoder_output)
            hypothesis_embeddings.append(h_emb)
            
        hypothesis_embeddings = torch.stack(hypothesis_embeddings, dim=1)  # (B, num_hyps, D)
        
        # Step 3: Judger selects best hypothesis
        selected_idx, confidence, pruned_count = self.judger(
            encoder_output, hypothesis_embeddings
        )
        
        # Step 4: Decode selected hypothesis token to text
        hypothesis_text = self._decode_token(hypotheses[selected_idx])
        
        # Return all hypotheses for transparency
        all_hypotheses = list(zip(
            [self._decode_token(h) for h in hypotheses],
            confidences
        ))
        
        return hypothesis_text, confidence, pruned_count, all_hypotheses
    
    def _get_hypothesis_embedding(self, head, encoder_output):
        """Re-run head to get embedding (used during training/inference)."""
        x = head.input_proj(encoder_output).unsqueeze(1)
        for layer in head.layers:
            x = layer(x)
        return x.squeeze(1)  # (B, D)
    
    def _decode_token(self, token_id):
        """Map token ID back to text."""
        # Implementation depends on tokenizer
        pass
```

---

## 3. Training Pipeline

### 3.1 Training Setup

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import wandb

class System2Dataset(Dataset):
    """Training dataset for System2.
    
    Expected format:
    - query: str — input query
    - hypotheses: list of str — candidate hypotheses
    - correct_idx: int — index of correct hypothesis
    - confidence: float — calibrated confidence target (if known)
    """
    
    def __init__(self, data_path, tokenizer, max_len=1024):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def _load_data(self, data_path):
        """Load dataset from JSON/CSV.
        
        Format:
        [
            {
                "query": "Is this ticket about billing?",
                "hypotheses": [
                    "Yes, the user mentions being charged twice.",
                    "Could be billing or service complaint.",
                    "Unclear — more context needed."
                ],
                "correct_idx": 0,
                "confidence": 0.87
            },
            ...
        ]
        """
        import json
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query
        encoded = self.tokenizer(
            item['query'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'hypotheses': item['hypotheses'],
            'correct_idx': item['correct_idx'],
            'confidence': item.get('confidence', 0.5)
        }
```

### 3.2 Training Loop (Phase 3: Head Fine-tuning)

```python
def train_heads(system2, dataloader, optimizer, device, epochs=5):
    """Fine-tune thought heads with RLCD (strict proper scoring rules).
    
    Objective: Maximize expected reward = -Brier score
    (honest probability reporting maximizes expected reward)
    
    Reference: Laya's RLCD training
    """
    system2.to(device)
    system2.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            correct_idx = batch['correct_idx'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            hypotheses = []
            confidences = []
            for head in system2.heads:
                h_token, h_confidence = head(
                    system2.encoder(input_ids, attention_mask)
                )
                hypotheses.append(h_token)
                confidences.append(h_confidence)
            
            # RLCD loss: Brier score on confidence scores
            # (honest confidence maximizes expected reward)
            brier_loss = 0
            for i, (h_token, h_confidence) in enumerate(zip(
                hypotheses, confidences
            )):
                is_correct = (i == correct_idx.item())
                # Brier score: (confidence - ground_truth)^2
                # where ground_truth = 1 if correct, 0 if incorrect
                ground_truth = 1.0 if is_correct else 0.0
                brier = (h_confidence - ground_truth) ** 2
                brier_loss += brier
            
            brier_loss /= len(hypotheses)
            brier_loss.backward()
            
            optimizer.step()
            total_loss += brier_loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return system2
```

### 3.3 Training Loop (Phase 4: Judger Training)

```python
def train_judger(system2, dataloader, optimizer, device, epochs=3):
    """Train judger layer to correctly score and select hypotheses.
    
    Objective: Select the hypothesis that maximizes calibrated accuracy.
    Training: RLCD on judger's scoring function.
    """
    system2.to(device)
    system2.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            correct_idx = batch['correct_idx'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (run all heads)
            encoder_output = system2.encoder(input_ids, attention_mask)
            
            hypothesis_embeddings = []
            for head in system2.heads:
                h_emb = system2._get_hypothesis_embedding(head, encoder_output)
                hypothesis_embeddings.append(h_emb)
            
            hypothesis_embeddings = torch.stack(hypothesis_embeddings, dim=1)
            
            # Judger selects best hypothesis
            selected_idx, confidence, _ = system2.judger(
                encoder_output, hypothesis_embeddings
            )
            
            # Loss: cross-entropy between selected and correct
            loss = nn.functional.cross_entropy(
                system2.judger.hypothesis_embeddings.mean(dim=-1),
                correct_idx
            )
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return system2
```

### 3.4 Training Loop (Phase 5: End-to-End Fine-tuning)

```python
def train_end_to_end(system2, dataloader, optimizer, device, epochs=5):
    """Full model fine-tuning via autoresearch loop.
    
    Loop protocol (Karpathy):
    1. Snapshot current state
    2. Propose head configuration (modify head weights)
    3. Judge on validation set
    4. Better → commit (keep new weights)
       Worse → revert (restore old weights)
    """
    system2.to(device)
    system2.train()
    
    best_weights = None  # snapshot of best configuration
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            correct_idx = batch['correct_idx'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            result = system2(input_ids, attention_mask)
            hypothesis_text, confidence, pruned_count, all_hypotheses = result
            
            # Compute loss (Brier score)
            selected_idx = all_hypotheses.index(
                all_hypotheses[correct_idx.item()]
            )
            brier = (confidence - 1.0 if selected_idx == correct_idx.item() else 0.0) ** 2
            brier.backward()
            
            optimizer.step()
            epoch_loss += brier.item()
        
        # Autoresearch loop: propose configuration → judge → commit/revert
        if should_propose_configuration(epoch):
            new_weights = propose_new_configuration(system2)
            val_loss = evaluate_on_validation(new_weights, system2.val_dataloader)
            
            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                best_weights = new_weights
            else:
                system2.load_state_dict(best_weights)  # revert
    
    return system2
```

### 3.5 Training Data Preparation

**Tier 1 datasets (reasoning):**

```python
def load_gsm8k():
    """Load GSM8K dataset (grade-school math word problems)."""
    import datasets
    dataset = datasets.load_dataset("gsm8k", "main")
    return dataset["train"], dataset["test"]

def load_math():
    """Load MATH dataset (competition-level math problems)."""
    import datasets
    dataset = datasets.load_dataset("HuggingFaceH4/MATH")
    return dataset["train"], dataset["test"]

def load_ai2_arc():
    """Load AI2 ARC dataset (grade-school science)."""
    import datasets
    dataset = datasets.load_dataset("allenai/ai2_arc")
    return dataset["train"], dataset["test"]
```

**Tier 2 datasets (classification):**

```python
def load_ag_news():
    """Load AG News dataset (news classification)."""
    import datasets
    dataset = datasets.load_dataset("ag_news")
    return dataset["train"], dataset["test"]

def load_banking77():
    """Load Banking77 dataset (customer service intents)."""
    import datasets
    dataset = datasets.load_dataset("PragmaticWavertree/Banking77")
    return dataset["train"], dataset["test"]
```

**Synthetic data generation:**

```python
def generate_synthetic_data(base_model, queries, num_hypotheses_per_query=10):
    """Generate synthetic training data using a base model.
    
    Caveat: Synthetic data carries the base model's biases.
    Validate against held-out ground-truth sets.
    
    Reference: "Self-Alignment" papers
    """
    import requests
    
    synthetic_data = []
    for query in queries:
        # Generate N hypotheses via base model
        hypotheses = []
        for i in range(num_hypotheses_per_query):
            prompt = f"Problem: {query}\nProvide a possible interpretation."
            response = base_model.generate(prompt, temperature=0.3 + 0.1*i)
            hypotheses.append(response["choices"][0]["message"]["content"])
        
        # Label (if ground truth available)
        synthetic_data.append({
            "query": query,
            "hypotheses": hypotheses,
            "correct_idx": 0  # placeholder — label manually or auto-label
        })
    
    return synthetic_data
```

---

## 4. Inference Pipeline

### 4.1 Basic Inference

```python
import torch
from transformers import AutoTokenizer

class System2Inference:
    """Run System2 model for inference."""
    
    def __init__(self, model_path, tokenizer_name="Qwen3Tokenizer", 
                 device="mps"):  # "mps" for Apple Silicon
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model
        self.model = System2(num_heads=10, vocab_size=151000)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
    def infer(self, query, top_k_hypotheses=5):
        """
        Args:
            query: str — input query
            top_k_hypotheses: int — number of hypotheses to return
            
        Returns:
            (selected_hypothesis, confidence_score, all_hypotheses)
        """
        # Tokenize
        encoded = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            hypothesis_text, confidence, pruned_count, all_hypotheses = \
                self.model(input_ids, attention_mask)
        
        return hypothesis_text, confidence, pruned_count, all_hypotheses
```

### 4.2 Example Usage

```python
# Initialize system
inference = System2Inference(
    model_path="./models/system2_checkpoint.pt",
    device="mps"
)

# Query
query = "Is this customer service ticket about billing?"

# Inference
hypothesis, confidence, pruned_count, all_hypotheses = inference.infer(
    query, top_k_hypotheses=5
)

print(f"Selected: {hypothesis}")
print(f"Confidence: {confidence:.2%}")
print(f"Pruned: {pruned_count} hypotheses")
print(f"\nAll hypotheses:")
for h, c in all_hypotheses:
    print(f"  [{c:.0%}] {h}")
```

---

## 5. Repository Structure

```
system2/
├── 01-system-architecture.md    # This document (architecture)
├── 02-training-plan.md          # This document (training)
├── 03-implementation-spec.md    # This document (implementation)
├── src/
│   ├── __init__.py
│   ├── model.py                 # System2, Encoder, ThoughtHead, Judger classes
│   ├── dataset.py               # System2Dataset class
│   ├── train.py                 # Training scripts (Phases 1-5)
│   ├── infer.py                 # Inference script (System2Inference)
│   └── utils.py                 # Helper functions
├── data/
│   ├── gsm8k/                   # GSM8K dataset
│   ├── math/                    # MATH dataset
│   ├── ai2_arc/                 # AI2 ARC dataset
│   ├── ag_news/                 # AG News dataset
│   ├── banking77/               # Banking77 dataset
│   └── synthetic/               # Synthetic training data
├── models/
│   ├── system2_checkpoint.pt    # Trained model checkpoint
│   └── encoder_checkpoint.pt    # Pre-trained encoder checkpoint
├── configs/
│   ├── head_config.json         # Thought head configurations
│   ├── eval_config.json         # Evaluation settings
│   └── train_config.json        # Training configurations
├── experiments/                 # Logged training runs (W&B sync)
│   ├── experiment_001/
│   ├── experiment_002/
│   └── ...
└── tests/
    ├── test_model.py            # Unit tests
    ├── test_inference.py        # Inference tests
    └── test_training.py         # Training tests
```

---

## 6. Dependencies

```
# requirements.txt
torch>=2.1.0
transformers>=4.35.0
datasets>=2.14.0
wandb>=0.16.0
bitsandbytes>=0.42.0  # 4-bit quantization
mlx>=0.12.0  # Apple Silicon optimization (optional)
```

**Apple Silicon specific:**

```bash
# Install MLX (Apple's ML framework)
pip install mlx

# Install Apple Silicon optimization (via Splash or MTPLX)
brew install incoai/tap/splash  # Splash runtime
# or
brew install youssofal/mtplx/mtplx  # MTPLX runtime
```

---

## 7. Benchmarking

### 7.1 Evaluation Metrics

**Accuracy:**
```bash
# Run GSM8K benchmark
python -c "
from system2.inference import System2Inference
from datasets import load_dataset

model = System2Inference('./models/system2_checkpoint.pt')
dataset = load_dataset('gsm8k', 'main')

correct = 0
total = 0
for item in dataset['test']:
    hypothesis, confidence, _, _ = model.infer(item['question'])
    if hypothesis.strip() == item['answer'].strip():
        correct += 1
    total += 1

print(f'GSM8K Accuracy: {correct/total:.2%}')
"
```

**Calibration:**
```python
def evaluate_calibration(predictions, ground_truths):
    """Evaluate model's calibration (ECE, Brier score).
    
    Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
    """
    import numpy as np
    
    confidences = [p["confidence"] for p in predictions]
    correct = [1 if p["correct"] else 0 for p in predictions]
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if bin_mask.sum() > 0:
            bin_confidence = np.array(confidences)[bin_mask].mean()
            bin_accuracy = np.array(correct)[bin_mask].mean()
            ece += bin_mask.sum() / len(confidences) * abs(bin_confidence - bin_accuracy)
    
    # Brier score
    brier = np.mean([(c - g) ** 2 for c, g in zip(confidences, correct)])
    
    return {"ece": ece, "brier": brier}
```

**Latency:**
```bash
# Measure latency
python -c "
import time
from system2.inference import System2Inference

model = System2Inference('./models/system2_checkpoint.pt')
queries = ['Is this about billing?', 'What is the capital of France?', ...]

latencies = []
for query in queries:
    start = time.time()
    hypothesis, confidence, _, _ = model.infer(query)
    latency = time.time() - start
    latencies.append(latency)

print(f'Average latency: {sum(latencies)/len(latencies)*1000:.0f}ms')
print(f'P50 latency: {sorted(latencies)[len(latencies)//2]*1000:.0f}ms')
print(f'P99 latency: {sorted(latencies)[int(len(latencies)*0.99)]*1000:.0f}ms')
"
```

### 7.2 Benchmark Datasets

| Dataset | Size | Task | Source |
|---|---|---|---|
| GSM8K | 8.5K | Grade-school math | [GSM8K](https://huggingface.co/datasets/gsm8k) |
| MATH | 13K | Competition-level math | [MATH](https://huggingface.co/datasets/HuggingFaceH4/MATH) |
| AI2 ARC | 7.8K | Grade-school science | [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) |
| AG News | 120K | News classification | [AG News](https://huggingface.co/datasets/ag_news) |
| Banking77 | 15K | Customer service | [Banking77](https://huggingface.co/datasets/PragmaticWavertree/Banking77) |

---

## 8. Risks and Mitigations

### 8.1 Training Risks

**Risk:** Encoder pre-training takes too long (~1 week on M4 Pro).
**Mitigation:** Use a pre-trained ModernBERT-large checkpoint (available on Hugging Face). Skip pre-training entirely and fine-tune only heads + judger.

**Risk:** Head fine-tuning doesn't converge.
**Mitigation:** Start with fewer heads (3-5) and scale up. Use learning rate scheduling (warmup + cosine decay). Use gradient clipping.

**Risk:** Judger doesn't learn to select correctly.
**Mitigation:** Train judger separately (freeze heads, train only judger). Use RLCD (strict proper scoring rules) for calibrated confidence.

### 8.2 Inference Risks

**Risk:** High latency (>500ms per query).
**Mitigation:** Reduce number of heads (5 instead of 10). Use 4-bit quantization. Cache encoder outputs for repeated queries.

**Risk:** Poor calibration (confidence doesn't match accuracy).
**Mitigation:** Train with RLCD (strict proper scoring rules). Validate calibration on held-out test sets. Adjust confidence threshold dynamically.

### 8.3 Hardware Risks

**Risk:** Model doesn't fit in RAM.
**Mitigation:** Use 4-bit quantization (reduces model size by 4×). Use MLX for Apple Silicon optimization (offloads to GPU).

**Risk:** Training takes too long (weeks on M4 Pro).
**Mitigation:** Use fewer training epochs (start with 2-3). Use smaller datasets (GSM8K is sufficient for v1). Use gradient accumulation to simulate larger batch sizes.

---

## 9. Milestones

### Milestone 1: Basic System (Week 1-2)

- [ ] Install dependencies (torch, transformers, datasets, bitsandbytes)
- [ ] Load pre-trained ModernBERT-large (skip pre-training)
- [ ] Implement ThoughtHead class (single head)
- [ ] Test inference with 1 head on GSM8K test set
- **Success criterion:** System produces reasonable hypotheses with calibrated confidence

### Milestone 2: Multi-Head System (Week 3-4)

- [ ] Implement 5-10 parallel thought heads
- [ ] Implement Judger class (cross-attention + self-attention)
- [ ] Train heads with RLCD (Phase 3)
- [ ] Test on GSM8K and AG News test sets
- **Success criterion:** Multi-head system outperforms single-head system

### Milestone 3: Judger Training (Week 5)

- [ ] Train judger layer (Phase 4)
- [ ] Test judger's selection accuracy on held-out test set
- **Success criterion:** Judger correctly selects best hypothesis >80% of time

### Milestone 4: End-to-End System (Week 6-7)

- [ ] Train all components end-to-end (Phase 5)
- [ ] Evaluate on GSM8K, MATH, AG News, Banking77
- [ ] Measure accuracy, calibration, latency
- **Success criterion:** Calibrated accuracy >75% on GSM8K, <500ms latency

### Milestone 5: Prototype v1 (Week 8)

- [ ] Package system2 as a Python library
- [ ] Write documentation and examples
- [ ] Publish to Hugging Face (if ready)
- **Success criterion:** Runnable system that works on a laptop

---

## 10. Future Work

### 10.1 Scale Up

**Larger models:** Scale encoder to 1.3B parameters (ModernBERT-large → 1.3B). Scale heads to 20-30. This will require ~8GB RAM and 2-3× training time.

**Longer context:** Extend encoder's context length from 1024 to 8192 tokens (Laya's `laya-multilingual` supports up to 8192 tokens). This requires more memory but allows longer queries.

**More hypotheses:** Increase number of heads to 20-30. Measure accuracy vs. latency tradeoff.

### 10.2 Text Generation

**Add generation head:** Currently, heads produce token indices (hypotheses). Add a text generation head that converts hypotheses to natural language:

```python
class GenerationHead(nn.Module):
    """Convert hypothesis token to natural language text.
    
    Architecture:
    - Input: hypothesis embedding (from thought head)
    - Output: token-by-token text generation (autoregressive)
    - Parameters: ~50M (small decoder)
    
    Note: This adds autoregressive generation, which is a departure
    from the pure System 2 architecture. Use selectively.
    """
    
    def __init__(self, vocab_size=151000, hidden_size=512):
        super().__init__()
        self.generation = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8),
            num_layers=4
        )
        self.output = nn.Linear(hidden_size, vocab_size)
```

### 10.3 Dynamic Head Selection

**Adaptive number of heads:** Instead of a fixed number of heads, dynamically adjust based on query complexity:

```python
def dynamic_head_count(query, encoder_output):
    """Select number of heads based on query complexity.
    
    Simple query: 3 heads (fast)
    Complex query: 10 heads (accurate)
    
    Implementation: Train a small classifier (5M params) to predict
    query complexity from encoder output. Output: number of heads.
    """
    pass
```

---

## 11. References

- Selsam, D. (2026). "Personal Statement on AI Risk." OpenAI.
- Kahneman, D. (2011). *Thinking, Fast and Slow*.
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."
- Shoeybi, M., et al. (2024). "ModernBERT: Modernized BERT Architecture." Hugging Face.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks."
- Laya documentation. ConvAI Innovations. https://huggingface.co/convaiinnovations/laya
- Splash documentation. inco.ai. https://github.com/incoai/splash
- MTPLX documentation. https://github.com/youssofal/MTPLX
- Karpathy's autoresearch. https://github.com/tj-doomer-it/test-training
