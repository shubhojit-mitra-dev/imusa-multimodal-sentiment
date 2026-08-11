# IMUSA Deep Dive: Understanding Everything From Scratch

> A complete guide for a backend/devops engineer entering ML — covering what the task is, how multimodal AI works, and how we'll turn it into a production-grade distributed systems project.

---

## Part 1: What Exactly Is This Task?

### The One-Liner
You're given **3,502 Punjabi memes** (images with text on them). Each meme has a label: **Sarcasm**, **Neutral**, **Offensive**, or **Motivational**. Your job is to build an AI model that looks at a new, unseen meme and correctly predicts which of these 4 categories it belongs to.

### Think of It Like This (Backend Analogy)
Imagine you're building a **request classifier** for an API gateway. Incoming requests (memes) need to be routed to one of 4 queues (Sarcasm / Neutral / Offensive / Motivational). You can't write `if-else` rules because the "logic" is too nuanced — a meme might say something positive but mean it sarcastically. So instead, you train a model to learn the patterns from 3,002 labeled examples, and then it classifies the remaining 500 test memes it has never seen.

### Why Is This Hard?

```
┌─────────────────────────────────────────────────────┐
│                  A MEME HAS TWO PARTS               │
│                                                     │
│   ┌─────────────┐        ┌─────────────────────┐   │
│   │             │        │                     │   │
│   │   IMAGE     │   +    │   TEXT (Punjabi)     │   │
│   │  (visual)   │        │   (textual)         │   │
│   │             │        │                     │   │
│   └─────────────┘        └─────────────────────┘   │
│                                                     │
│   Neither alone tells you the sentiment.            │
│   You need BOTH. That's why it's "multimodal".      │
└─────────────────────────────────────────────────────┘
```

1. **Multimodal = Multiple types of data.** A meme's meaning comes from the **interplay** of the image and the text. A picture of someone crying + "Best day ever" = **Sarcasm**. The same picture + "I miss you" = could be **Neutral/Motivational**. You can't just look at the image or just read the text.

2. **Low-resource language.** Punjabi (written in Gurmukhi script: ਪੰਜਾਬੀ) doesn't have the massive pre-trained models that English has. English has GPT-4, BERT, RoBERTa all trained on billions of words. Punjabi? Far fewer resources. So you can't just plug into a pre-built pipeline.

3. **Cultural context.** Sarcasm in Punjabi memes relies on cultural references, inside jokes, Bollywood, politics, etc. A model trained only on English memes wouldn't understand these nuances.

4. **Class imbalance.** In real-world social media data, you might have 1,000 Neutral memes but only 200 Offensive ones. The model might just learn to always say "Neutral" and still get 60% accuracy — but it would be useless for the classes that matter most.

---

## Part 2: Machine Learning Concepts You Need (Explained for Backend Engineers)

### 2.1 What Is a "Model"?

A model is essentially a **mathematical function with millions of tunable parameters** (numbers). 

```
Backend analogy:
    - A function: f(input) → output
    - The "parameters" are like configuration knobs
    - "Training" = automatically tuning those knobs using data
    - "Inference" = running the tuned function on new data
```

Before training: `f(meme) → random garbage`  
After training: `f(meme) → "Sarcasm"` (hopefully correct)

### 2.2 What Is Training?

Training is an optimization loop. Here's the pseudocode a backend engineer would understand:

```python
# Pseudocode — this is literally what happens
model = initialize_random_model()

for epoch in range(num_epochs):  # repeat N times over full dataset
    for batch in training_data:  # process data in chunks
        images, texts, labels = batch  # the meme data

        predictions = model(images, texts)  # forward pass: model guesses
        loss = compute_error(predictions, labels)  # how wrong was it?

        gradients = compute_gradients(loss)  # which parameters caused the error?
        model.update_parameters(gradients, learning_rate)  # fix them slightly

# After training:
prediction = model(new_unseen_meme)  # inference
```

| ML Term | Backend Equivalent |
|---|---|
| **Epoch** | One full pass through the entire dataset (like processing all messages in a Kafka topic) |
| **Batch** | A chunk of data processed at once (like batch size in a message queue consumer) |
| **Loss** | Error metric — how wrong the model is (like an error rate in monitoring) |
| **Gradient** | The direction + magnitude to adjust each parameter (like a feedback signal) |
| **Learning Rate** | Step size for parameter updates (like a rate limiter — too fast = unstable, too slow = never converges) |
| **Overfitting** | Model memorizes training data but fails on new data (like hardcoding responses instead of building logic) |
| **Inference** | Using the trained model to make predictions (like calling your deployed service) |

### 2.3 What Is "Multimodal" Learning?

In our case, each meme has **two modalities** (types of input):

```
Modality 1: IMAGE  →  What does the picture look like?
Modality 2: TEXT   →  What does the Punjabi text say?
```

The challenge is: **how do you combine information from an image and text into a single prediction?**

This is called **fusion**, and there are three main strategies:

```
┌─────────────────────────────────────────────────┐
│           EARLY FUSION                          │
│                                                 │
│  Image features ──┐                            │
│                    ├──→ Concatenate → Classifier│
│  Text features  ──┘                            │
│                                                 │
│  Simple. Just smash them together early.        │
│  Like joining two DB tables before querying.    │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│           LATE FUSION                           │
│                                                 │
│  Image features → Image Classifier ──┐         │
│                                      ├──→ Vote │
│  Text features  → Text Classifier  ──┘         │
│                                                 │
│  Each modality makes its own decision,          │
│  then they vote. Like microservices.            │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│         CROSS-ATTENTION FUSION (SOTA)           │
│                                                 │
│  Image features ←──────→ Text features          │
│        ↕ attend to each other ↕                 │
│  Fused representation → Classifier              │
│                                                 │
│  The image "looks at" the text and vice versa.  │
│  Like a join with a WHERE clause that adapts.   │
│  This is what state-of-the-art models use.      │
└─────────────────────────────────────────────────┘
```

### 2.4 Transfer Learning (The Most Important Concept)

You do NOT train a model from absolute zero. That would require millions of memes and weeks of GPU time.

Instead, you use **pre-trained models** — models that someone else already trained on massive datasets — and **fine-tune** them on your specific task.

```
Backend analogy:
    You don't write a web server from scratch.
    You take Nginx/Express/FastAPI and configure it for your use case.
    
    Similarly:
    You take a pre-trained vision model (knows how to "see")
    You take a pre-trained text model (knows how to "read")
    You fine-tune them on your 3,002 Punjabi memes
```

**For images**, we'll use models like:
- **ViT (Vision Transformer)** or **ResNet** — pre-trained on millions of images (ImageNet). They already know what faces, objects, emotions look like.
- **CLIP (by OpenAI)** — pre-trained to understand the *relationship* between images and text. Very powerful for memes.

**For text**, we'll use models like:
- **XLM-RoBERTa** — a multilingual transformer pre-trained on 100+ languages, including Punjabi
- **MuRIL** — specifically designed for Indian languages by Google
- **IndicBERT** — trained specifically on Indic languages

### 2.5 What Is OCR and Why Do We Need It?

The text in memes is **embedded in the image** — it's not a separate text file. So we need **OCR (Optical Character Recognition)** to extract the Punjabi text from the image.

```
┌──────────────────────┐       ┌──────────────────┐
│ Meme Image           │       │ Extracted Text   │
│ ┌──────────────────┐ │  OCR  │                  │
│ │ ਕੀ ਹਾਲ ਹੈ       │ │ ────→ │ "ਕੀ ਹਾਲ ਹੈ      │
│ │ ਬਈ? 😂          │ │       │  ਬਈ?"            │
│ └──────────────────┘ │       │                  │
│   (person laughing)  │       │                  │
└──────────────────────┘       └──────────────────┘
```

> [!NOTE]
> The dataset might already provide the text separately. If so, OCR isn't needed during training. But for production inference (when someone uploads a new meme), you'll definitely need OCR in your pipeline.

Tools: **EasyOCR** (supports Punjabi, deep-learning based, better for noisy meme images) or **Tesseract** (lighter, faster, but less accurate on memes).

---

## Part 3: The Model Architecture (What We'll Build)

Here's the high-level architecture of our multimodal classifier:

```
                        ┌─────────────┐
                        │  INPUT:     │
                        │  Meme Image │
                        └──────┬──────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              │              ▼
    ┌───────────────────┐      │    ┌──────────────────┐
    │  IMAGE ENCODER    │      │    │   OCR ENGINE     │
    │  (ViT / ResNet /  │      │    │  (EasyOCR or     │
    │   CLIP Vision)    │      │    │   provided text)  │
    │                   │      │    │                  │
    │  Pre-trained,     │      │    └────────┬─────────┘
    │  fine-tuned       │      │             │
    └────────┬──────────┘      │             ▼
             │                 │    ┌──────────────────┐
             │                 │    │  TEXT ENCODER     │
             │                 │    │  (XLM-RoBERTa /  │
             │                 │    │   MuRIL /         │
             │                 │    │   IndicBERT)      │
             │                 │    │                  │
             │                 │    │  Pre-trained,    │
             │                 │    │  fine-tuned      │
             │                 │    └────────┬─────────┘
             │                 │             │
             ▼                 │             ▼
    ┌─────────────┐            │    ┌─────────────┐
    │ Image       │            │    │ Text        │
    │ Features    │            │    │ Features    │
    │ (vector of  │            │    │ (vector of  │
    │  768 nums)  │            │    │  768 nums)  │
    └──────┬──────┘            │    └──────┬──────┘
           │                   │           │
           └───────────┬───────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  FUSION LAYER   │
              │                 │
              │  Cross-attention│
              │  or concatenate │
              │  + MLP layers   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  CLASSIFIER     │
              │  (Linear layer) │
              │                 │
              │  4 outputs:     │
              │  [0.8, 0.1,     │
              │   0.05, 0.05]   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  PREDICTION:    │
              │  "Sarcasm" ✓    │
              └─────────────────┘
```

### What each part does:

| Component | What it does | Backend analogy |
|---|---|---|
| **Image Encoder** | Converts a meme image (pixels) into a dense vector of numbers (features) that capture what's in the image | Like serializing a complex object into a compact representation |
| **OCR / Text Input** | Extracts or receives the Punjabi text from the meme | Like parsing a request body |
| **Text Encoder** | Converts Punjabi text into a dense vector of numbers that capture meaning | Like hashing a string into a fixed-size embedding |
| **Fusion Layer** | Combines image + text features into a single unified representation | Like a JOIN operation that merges two data streams |
| **Classifier** | Takes the fused representation and outputs probabilities for each class | Like a router that decides which queue to send to |

---

## Part 4: The Full Project Vision (This Is Where Your Skills Shine)

This is where we go **far beyond** just training a model. Here's the complete system we'll build:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE SYSTEM ARCHITECTURE                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      KUBERNETES CLUSTER                            │ │
│  │                                                                     │ │
│  │  ┌───────────────────────────────────────────────┐                 │ │
│  │  │  TRAINING PLANE (PyTorch Distributed)         │                 │ │
│  │  │                                               │                 │ │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │                 │ │
│  │  │  │Worker 0 │  │Worker 1 │  │Worker 2 │      │                 │ │
│  │  │  │(master) │←→│         │←→│         │      │                 │ │
│  │  │  │  GPU    │  │  GPU    │  │  GPU    │      │                 │ │
│  │  │  └────┬────┘  └────┬────┘  └────┬────┘      │                 │ │
│  │  │       │            │            │            │                 │ │
│  │  │       └────────────┼────────────┘            │                 │ │
│  │  │                    │                         │                 │ │
│  │  │              ┌─────▼─────┐                   │                 │ │
│  │  │              │  NCCL     │                   │                 │ │
│  │  │              │  Comms    │                   │                 │ │
│  │  │              └───────────┘                   │                 │ │
│  │  └───────────────────────────────────────────────┘                 │ │
│  │                         │                                          │ │
│  │                    Model Artifacts                                 │ │
│  │                         │                                          │ │
│  │                    ┌────▼─────┐                                    │ │
│  │                    │  MLflow  │ ◄── Experiment Tracking            │ │
│  │                    │  Server  │     Model Registry                 │ │
│  │                    └────┬─────┘                                    │ │
│  │                         │                                          │ │
│  │  ┌──────────────────────▼────────────────────────┐                │ │
│  │  │  SERVING PLANE                                │                │ │
│  │  │                                               │                │ │
│  │  │  ┌──────────────┐    ┌──────────────┐        │                │ │
│  │  │  │  KServe /    │    │  API Gateway │        │                │ │
│  │  │  │  FastAPI     │◄───│  (Ingress)   │◄── Requests            │ │
│  │  │  │  Model Server│    │              │        │                │ │
│  │  │  └──────────────┘    └──────────────┘        │                │ │
│  │  └───────────────────────────────────────────────┘                │ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │  APPLICATION PLANE                                          │   │ │
│  │  │                                                             │   │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌────────────┐ │   │ │
│  │  │  │ Backend  │  │PostgreSQL│  │  Redis   │  │ Frontend   │ │   │ │
│  │  │  │ (FastAPI)│←→│  (DB)    │  │ (Cache/  │  │ (Next.js)  │ │   │ │
│  │  │  │          │  │          │  │  Queue)  │  │            │ │   │ │
│  │  │  └──────────┘  └──────────┘  └─────────┘  └────────────┘ │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │  OBSERVABILITY                                              │   │ │
│  │  │  Prometheus + Grafana + TensorBoard                         │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────┐                                       │
│  │  STORAGE (S3 / MinIO)       │                                       │
│  │  - Training data            │                                       │
│  │  - Model checkpoints        │                                       │
│  │  - Model artifacts          │                                       │
│  └─────────────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.1 The Five Pillars of This Project

#### Pillar 1: 🧠 ML Model (The Core)
This is the multimodal classifier described in Part 3. You'll:
- Preprocess meme images + extract/receive Punjabi text
- Fine-tune pre-trained encoders (ViT + XLM-RoBERTa)
- Implement a fusion mechanism
- Train, evaluate, and optimize

#### Pillar 2: 🏋️ Distributed Training (Your Distributed Systems Expertise)
Instead of training on a single GPU, you'll use **PyTorch Distributed Data Parallel (DDP)** across multiple workers on Kubernetes.

**Why distributed training matters:**
```
Single GPU Training:
    1 GPU processes 32 memes per batch
    Time per epoch: ~10 minutes

Distributed Training (4 GPUs):
    Each GPU processes 32 memes (128 total per step)
    Each GPU has a COPY of the model
    After each step, GPUs synchronize gradients via NCCL
    Time per epoch: ~2.5 minutes (near-linear speedup)
```

**How DDP works (think of it like a distributed database):**

```
┌────────────────────────────────────────────────────────┐
│            PyTorch DDP — Conceptual Flow               │
│                                                        │
│  1. Each worker (GPU) gets a FULL COPY of the model    │
│     (like replicas in a database cluster)               │
│                                                        │
│  2. Training data is SHARDED across workers             │
│     (like partitioning in Kafka or Cassandra)           │
│     Worker 0 gets memes 0-749                          │
│     Worker 1 gets memes 750-1499                       │
│     Worker 2 gets memes 1500-2249                      │
│     Worker 3 gets memes 2250-3001                      │
│                                                        │
│  3. Each worker computes gradients on its own shard     │
│                                                        │
│  4. ALL-REDUCE: Workers exchange and average gradients  │
│     (like a consensus protocol — everyone agrees on     │
│      the same update)                                   │
│                                                        │
│  5. Each worker applies the SAME averaged gradient      │
│     → All models stay in sync (strong consistency)      │
│                                                        │
│  6. Repeat until converged                              │
└────────────────────────────────────────────────────────┘
```

**On Kubernetes**, you'll use:
- **Kubeflow Training Operator** → Defines `PyTorchJob` CRDs that manage worker pods
- **NCCL** → The communication backend (like gRPC for GPUs)
- **torchrun** → PyTorch's built-in distributed launcher
- **Checkpointing to S3/MinIO** → Fault tolerance (if a pod dies, resume from last checkpoint)

#### Pillar 3: 🔄 MLOps Pipeline
This is where you track experiments, version models, and automate the lifecycle:

| Tool | Purpose | Backend Equivalent |
|---|---|---|
| **MLflow** | Track experiments (hyperparameters, metrics, artifacts) + model registry | Like Datadog + Docker Registry |
| **TensorBoard** | Visualize training curves (loss, accuracy over time) | Like Grafana for ML metrics |
| **DVC** (optional) | Version control for datasets (too big for git) | Like Git LFS for data |
| **Docker** | Containerize everything | You know this one 😄 |
| **Helm charts** | Package K8s deployments | You know this one too |

#### Pillar 4: 🌐 Backend + API
A production API that accepts meme images and returns predictions:

```
POST /api/v1/predict
Content-Type: multipart/form-data

Body: { image: <meme.jpg> }

Response:
{
    "prediction": "Sarcasm",
    "confidence": 0.87,
    "probabilities": {
        "sarcasm": 0.87,
        "neutral": 0.06,
        "offensive": 0.04,
        "motivational": 0.03
    },
    "extracted_text": "ਕੀ ਹਾਲ ਹੈ ਬਈ?",
    "processing_time_ms": 142
}
```

Stack:
- **FastAPI** (Python) — REST API + WebSocket for async predictions
- **PostgreSQL** — User management, prediction history, quotas
- **Redis** — Caching, rate limiting, async job queue
- **Celery/ARQ** — Background workers for inference (meme preprocessing + model inference can take time)

#### Pillar 5: 🎨 Frontend Dashboard
A web dashboard where users can:
- Upload a meme and see the prediction with confidence scores
- View prediction history
- See API usage / quotas
- Visualize model performance metrics

Stack: **Next.js + React + Tailwind CSS**

### 4.2 Free Deployment Options

| Component | Free Option |
|---|---|
| **Model Hosting** | Hugging Face Spaces (free CPU tier, ZeroGPU for limited GPU) |
| **Model Inference API** | Hugging Face Serverless Inference API (rate-limited) |
| **Backend API** | Railway.app free tier / Render.com / Fly.io |
| **Frontend** | Vercel (free for personal projects) |
| **Database** | Supabase (free PostgreSQL) or PlanetScale |
| **K8s Cluster** | Minikube (local) or free credits from GCP/AWS |
| **MLflow** | Self-hosted on the same cluster |
| **Object Storage** | MinIO (self-hosted, S3-compatible) |

---

## Part 5: The Procedure — How We'll Actually Do This

### Phase 0: Foundation (Now — before dataset arrives)
- [  ] Set up project structure and development environment
- [  ] Understand the evaluation metrics (Accuracy, Precision, Recall, F1)
- [  ] Set up Docker + K8s development environment
- [  ] Research and choose pre-trained models (freeze these decisions)

### Phase 1: Data & Exploration (When dataset arrives)
- [  ] Load and explore the dataset (class distribution, image sizes, text lengths)
- [  ] Build the data preprocessing pipeline (image transforms, text tokenization)
- [  ] Implement OCR pipeline if text isn't provided separately
- [  ] Create train/validation split (stratified to maintain class balance)
- [  ] Handle class imbalance (weighted loss, oversampling, augmentation)

### Phase 2: Model Development (Local, single GPU or CPU)
- [  ] Implement image encoder (ViT or CLIP vision encoder)
- [  ] Implement text encoder (XLM-RoBERTa or MuRIL)
- [  ] Implement fusion mechanism
- [  ] Implement classifier head
- [  ] Write training loop with PyTorch
- [  ] Set up MLflow experiment tracking
- [  ] Train locally on a small subset to verify everything works
- [  ] Iterate on hyperparameters and architecture

### Phase 3: Distributed Training on K8s
- [  ] Adapt training code for PyTorch DDP
- [  ] Dockerize the training job
- [  ] Write Kubeflow PyTorchJob manifest
- [  ] Set up MinIO for checkpoint storage
- [  ] Deploy and run distributed training
- [  ] Monitor with TensorBoard + Grafana

### Phase 4: Model Serving & API
- [  ] Export best model checkpoint
- [  ] Register model in MLflow Model Registry
- [  ] Build FastAPI inference service
- [  ] Deploy model server on K8s (or Hugging Face Spaces)
- [  ] Add Redis caching, rate limiting
- [  ] Write comprehensive API tests

### Phase 5: Frontend & Polish
- [  ] Build Next.js dashboard
- [  ] Implement meme upload + prediction visualization
- [  ] Add user auth, history, quotas
- [  ] Deploy frontend to Vercel

### Phase 6: Production Hardening
- [  ] CI/CD pipeline (GitHub Actions)
- [  ] Monitoring & alerting (Prometheus + Grafana)
- [  ] Load testing the inference endpoint
- [  ] Documentation (README, API docs, architecture diagrams)

---

## Part 6: Why This Project Is Resume Gold

### For ML/AI roles:
- ✅ Built a multimodal model (not just text or just images — both)
- ✅ Handled a low-resource language (shows you can work beyond English NLP)
- ✅ Transfer learning with state-of-the-art architectures
- ✅ Proper evaluation with F1, precision, recall on imbalanced data

### For Backend/Infrastructure roles:
- ✅ Distributed training on Kubernetes with PyTorch DDP
- ✅ Full MLOps pipeline (MLflow, model registry, experiment tracking)
- ✅ Production API with rate limiting, caching, async processing
- ✅ Containerized everything with Docker + Helm

### For Full-Stack/Product roles:
- ✅ End-to-end product: from model → API → frontend dashboard
- ✅ User management, quotas, prediction history
- ✅ Clean UI for non-technical users

### What makes it stand out:
> Most ML projects on GitHub are just a Jupyter notebook with `model.fit()`. This one is a **full production system** with distributed training, MLOps, APIs, and a frontend. That's incredibly rare and will immediately set you apart.

---

## Part 7: Key Decisions We Need to Make Together

Before we start implementing, here are the decisions we need to discuss:

### Decision 1: Image Encoder
| Option | Pros | Cons |
|---|---|---|
| **CLIP ViT** | Pre-trained on image-text pairs (perfect for memes), excellent zero-shot | Larger model, more compute |
| **ViT (plain)** | Good general vision features, well-understood | Not specifically trained on text-image pairs |
| **ResNet-50** | Lightweight, fast training | Older architecture, may miss nuanced features |

### Decision 2: Text Encoder
| Option | Pros | Cons |
|---|---|---|
| **XLM-RoBERTa** | Supports 100+ languages including Punjabi, very robust | Large model |
| **MuRIL** | Google's model specifically for Indian languages | May have less community support |
| **IndicBERT** | Trained specifically on Indic languages | Smaller training corpus |

### Decision 3: Fusion Strategy
| Option | Pros | Cons |
|---|---|---|
| **Concatenation + MLP** | Simple, fast to implement, good baseline | May miss cross-modal interactions |
| **Cross-Attention** | State-of-the-art, models interplay between modalities | More complex, needs more data |
| **CLIP joint embedding** | If using CLIP, image+text are already in same space | Less flexible for fine-tuning |

### Decision 4: Where to run distributed training
| Option | Pros | Cons |
|---|---|---|
| **Local Minikube + CPU** | Free, good for learning | Very slow training |
| **Cloud K8s (GKE/EKS) free credits** | Real GPUs, realistic distributed setup | Credits run out |
| **Google Colab + simulated DDP** | Free GPU, quick iteration | Not real K8s, limited to single node |

> [!IMPORTANT]
> We don't need to make these decisions right now. When the dataset arrives, we'll explore it first — the data itself will inform many of these choices.

---

## Part 8: Evaluation Metrics Explained

The competition evaluates on 4 metrics. Here's what they mean:

### Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)

If you classify 400 out of 500 test memes correctly → 80% accuracy

⚠️ Problem: If 80% of memes are "Neutral", a model that ALWAYS says 
   "Neutral" gets 80% accuracy but is completely useless.
```

### Precision (per class)
```
Precision for "Sarcasm" = 
    (Memes correctly classified as Sarcasm) / 
    (ALL memes the model SAID were Sarcasm)

"Of all the memes I labeled as Sarcasm, how many actually were?"
High precision = few false positives
```

### Recall (per class)
```
Recall for "Sarcasm" = 
    (Memes correctly classified as Sarcasm) / 
    (ALL memes that ACTUALLY ARE Sarcasm)

"Of all the actual Sarcasm memes, how many did I catch?"
High recall = few false negatives
```

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

The harmonic mean — balances precision and recall.
If either is low, F1 is low. You can't cheat by only optimizing one.

This is THE metric that matters most for imbalanced datasets.
```

> [!TIP]
> F1-Score will likely be the primary ranking metric. Focus your optimization efforts here. Techniques like class-weighted loss functions, stratified sampling, and data augmentation directly improve F1 on minority classes.

---

## Summary: What You Now Know

1. **The Task**: Classify 3,502 Punjabi memes into 4 sentiment categories using both image and text
2. **Why It's Hard**: Multimodal (image + text), low-resource language, cultural context, class imbalance
3. **The Model**: Pre-trained image encoder + pre-trained text encoder → fusion → classifier
4. **Training**: Back-propagation with gradient descent, measured by loss, evaluated by F1
5. **The Project**: A full production system with distributed training on K8s, MLOps, API, and frontend
6. **Your Edge**: Your backend/devops skills make the infrastructure parts trivial — most ML people can't do this
7. **Next Steps**: Wait for dataset, then explore it, and start building

**When the dataset arrives, we jump straight into action.** 🚀
