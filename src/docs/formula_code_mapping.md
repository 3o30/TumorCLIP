### Formula ↔ Code Mapping (TumorCLIP)

This document lists the key formulas mentioned in the discussion and provides the corresponding implementation locations in the codebase, along with brief notes to support review and reproducibility.

- **1) Multi-class Cross-Entropy**
  - Formula:
    L(theta) = - Σ_{i=1..C} y_i * log( softmax( f_theta(x) )[i] )
  - Code mapping:
```52:60:src/training/enhanced_single_modal_trainer.py
        criterion = nn.CrossEntropyLoss()
        optimizer = self.create_optimizer(model, optimizer_name, lr)
        scheduler = self.create_scheduler(optimizer, num_epochs=num_epochs)
```
  - Notes: The main training path uses `nn.CrossEntropyLoss()` (equivalent to the above when \(y\) is one-hot). If called as a function inside the model, it uses `F.cross_entropy` as well (see below).

- **2) Fusion composite loss (multi-task weighted loss)**
  - Formula:
    L_total = 0.5 · CE(s_fused, y) + 0.3 · L_focal(s_dense, y) + 0.2 · CE(s_clip, y)
  - Code mapping (notebook implementation):
```1230:1239:CLIP融合模型训练.ipynb
                fusion_loss = F.cross_entropy(fused_logits, labels)
                densenet_loss = model.densenet_branch.compute_loss(densenet_logits, labels, loss_type='focal')
                clip_loss = F.cross_entropy(clip_logits, labels)
                total_loss = 0.5 * fusion_loss + 0.3 * densenet_loss + 0.2 * clip_loss
```
  - Notes: The implementation combines three losses with weights 0.5/0.3/0.2. Note that `densenet_branch.compute_loss(..., loss_type='focal')` calls the custom focal implementation (below).

- **3) Focal Loss (DenseNet branch)**
  - Formula (paper form):
    FL(pt) = - α (1 - pt)^γ log(pt)
  - Code mapping:
```43:51:src/models/losses.py
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
```
  - Notes: The code first computes per-sample CE (no reduction), recovers \(p_t\) via \(p_t = \exp(-CE)\), weights by \((1-p_t)^\gamma\), and then applies the configured reduction—equivalent to the standard focal-loss form.

- **4) Text prototypes**
  - Formula:
    μ_k = (1 / N_k) Σ_{j=1..N_k} z_text_j
  - Code mapping (prompt definition & aggregation):
```1:16:src/config/constants.py
CLASS_NAMES = [
   "Glioma",
   ...
]
```
```414:435:CLIP融合模型训练.ipynb
            text_features = self.encode_prompts(class_prompts)
            prototype = text_features.mean(dim=0, keepdim=True)
            prototype = F.normalize(prototype, dim=-1)
            prototypes.append(prototype)
```
  - Notes: The code averages embeddings across prompts per class and explicitly applies L2 normalization (`F.normalize`), so the normalized \(\mu_k\) is used.

- **5) Image feature extraction & projection (DenseNet → CLIP embedding)**
  - Code mapping:
```180:195:src/models/densenet_variants.py
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        projected_features = self.feature_projection(features)  # [B, 512]
        return projected_features
```
  - Notes: This returns the projected image vector \(v\) (typically 512-D), which is later normalized in Tip-Adapter for similarity computation.

- **6) Adapter MLP (lightweight 2-layer 512→128→512) and trainable parameters**
  - Code mapping (notebook):
```807:813:CLIP融合模型训练.ipynb
            self.adapter = nn.Sequential(
                nn.Linear(cache_keys.shape[1], cache_keys.shape[1] // 4),
                nn.ReLU(),
                nn.Linear(cache_keys.shape[1] // 4, cache_keys.shape[1])
            )
```
```919:924:CLIP融合模型训练.ipynb
        params.extend(self.tip_adapter.get_adapter_params())
        params.extend(list(self.image_encoder.feature_projection.parameters()))
```
  - Notes: The adapter (512→128→512) and `image_encoder.feature_projection` parameters are added to the trainable parameter list, while the CLIP text encoder is frozen (not optimized).

- **7) Text-prototype similarity (text-prototype logits)**
  - Formula:
    s_text_k = cosine_similarity( v, μ_k )
  - Code mapping:
```836:851:CLIP融合模型训练.ipynb
        cache_keys_norm = F.normalize(self.cache_keys, dim=-1)
        similarities = torch.mm(adapted_features, cache_keys_norm.t())  # [B, N]
...
        clip_logits = torch.mm(image_features, self.clip_model.t())  # Assume clip_model is text prototypes
```
  - Notes: Vectors are normalized and cosine similarity is implemented via dot product; `clip_logits` are the similarity scores against text prototypes.

- **8) Cache retrieval weights & aggregation (Tip-Adapter KNN part)**
  - Formula:
    s_i = cosine_similarity( v, v_i )
    a_i = softmax( s_i / t_knn )
    s_cache_k = Σ_i a_i * y_i[k]
  - Code mapping:
```836:847:CLIP融合模型训练.ipynb
        cache_keys_norm = F.normalize(self.cache_keys, dim=-1)
        similarities = torch.mm(adapted_features, cache_keys_norm.t())  # [B, N]
        similarities = similarities / self.t_knn
        weights = F.softmax(similarities, dim=-1)  # [B, N]
        knn_logits = torch.mm(weights, self.cache_values)  # [B, C]
```
  - Notes: Matches the formula: normalized cosine similarity → temperature scaling → softmax weights \(a_i\) → weighted sum with the one-hot label matrix to obtain cache logits \(s_{\text{cache},k}\).

- **9) Tip-Adapter fusion (text + cache)**
  - Formula:
    s_tip_k = (1 − α) · s_text_k + α · s_cache_k
  - Code mapping:
```853:854:CLIP融合模型训练.ipynb
        final_logits = (1 - self.alpha) * clip_logits + self.alpha * knn_logits
```
  - Notes: \(\alpha\) is a hyperparameter (commonly 0.4–0.6) and the implementation directly combines the two evidence sources via a weighted sum.

- **10) Final fusion of DenseNet and Tip-Adapter (learnable weight)**
  - Formula:
    s_fused_k = (1 − w) · s_dense_k + w · s_tip_k,  w = sigmoid(u)（u 为可训练标量）
  - Code mapping:
```1019:1021:CLIP融合模型训练.ipynb
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
```
```1110:1114:CLIP融合模型训练.ipynb
            fusion_weight = torch.sigmoid(self.fusion_weight)  # Ensure weight is in 0-1
            fused_logits = (1 - fusion_weight) * densenet_logits + fusion_weight * clip_logits
```
  - Notes: A `sigmoid` maps an unconstrained parameter into (0,1) and fuses DenseNet logits with Tip-Adapter logits. Final prediction takes `argmax` over the fused logits.

- **11) Final prediction**
  - Formula:
    y_hat = argmax_k s_fused_k
  - Code mapping: In training/evaluation, predictions are obtained with `logits.argmax(dim=1)` on the fused logits (see the evaluation & saving section in the training notebook).

---
Notes & suggestions (brief)
- The implementation aligns with the math above. The main points to watch are scale control (e.g., whether to apply temperature to `clip_logits`) and dtype (under mixed precision, similarity distributions may shift). If needed, I can provide a patch to add a learnable temperature for `clip_logits` or normalize both logits to a comparable scale.

Author: auto-generated (per request)
Location: `src/docs/formula_code_mapping.md`


