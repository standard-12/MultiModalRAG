# Introduction to Artificial Intelligence and Machine Learning

## What is Artificial Intelligence?

Artificial Intelligence (AI) refers to the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, problem-solving, perception, and language understanding. AI systems are designed to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages.

Modern AI is primarily driven by machine learning techniques, where systems learn from data rather than being explicitly programmed with rules. This shift from rule-based systems to data-driven learning has enabled breakthroughs across countless domains.

## Machine Learning Fundamentals

Machine learning (ML) is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. Instead of writing rules by hand, we feed data to algorithms that discover patterns automatically.

### Supervised Learning

In supervised learning, models are trained on labelled data — each training example has an input and a correct output. The model learns a mapping from inputs to outputs. Common algorithms include:

- Linear and logistic regression
- Decision trees and random forests
- Support vector machines (SVM)
- Neural networks

Applications include spam detection, image classification, and medical diagnosis.

### Unsupervised Learning

Unsupervised learning finds hidden patterns in unlabelled data. The algorithm is not told what to look for — it discovers structure on its own. Key techniques include:

- Clustering (K-means, DBSCAN, hierarchical)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Autoencoders and generative models

Applications include customer segmentation, anomaly detection, and data compression.

### Reinforcement Learning

Reinforcement learning (RL) trains agents to make sequences of decisions by rewarding desired behaviour and penalising undesired behaviour. The agent interacts with an environment and learns a policy that maximises cumulative reward. Notable successes include game-playing agents (AlphaGo, Atari games) and robotics control.

## Deep Learning

Deep learning is a subfield of machine learning that uses neural networks with many layers (hence "deep"). These deep networks can learn hierarchical representations of data — for example, detecting edges → shapes → objects in images.

### Convolutional Neural Networks (CNNs)

CNNs are designed for processing grid-like data such as images. They use convolutional layers to scan across an image with learned filters, building up complex feature detectors. CNNs power modern image recognition, object detection, and video analysis.

### Recurrent Neural Networks (RNNs) and LSTMs

RNNs process sequences by maintaining a hidden state that is updated at each time step. Long Short-Term Memory (LSTM) networks add gating mechanisms to address the vanishing gradient problem, enabling the network to remember information over longer sequences. They are used for language modelling, translation, and speech recognition.

### Transformer Architecture

Transformers, introduced in the paper "Attention Is All You Need" (2017), rely entirely on self-attention mechanisms rather than recurrence. They process all tokens in parallel and capture long-range dependencies efficiently. Transformers are the foundation of large language models (LLMs) such as GPT-4, Claude, and Qwen.

Key components of a transformer:
- **Multi-head self-attention**: allows each token to attend to all other tokens
- **Positional encoding**: injects sequence order since attention is permutation-invariant
- **Feed-forward layers**: applied independently to each position
- **Layer normalisation** and **residual connections** for training stability

## Large Language Models (LLMs)

Large Language Models are transformer-based models trained on massive text corpora using self-supervised objectives such as next-token prediction. They develop emergent capabilities including code generation, reasoning, summarisation, translation, and question answering.

Popular open-weight LLMs that can run locally:
- **Llama 3** (Meta) — strong general-purpose model
- **Gemma 2** (Google DeepMind) — efficient, available in 2B and 9B sizes
- **Mistral / Mixtral** — strong reasoning, mixture-of-experts variant
- **Phi-3** (Microsoft) — small but highly capable
- **LLaVA / MoondreamVL** — multimodal models that understand images

## Retrieval-Augmented Generation (RAG)

RAG is a technique that grounds LLM responses in retrieved external knowledge. Instead of relying solely on the model's parametric memory (learned during training), RAG retrieves relevant documents from a knowledge base at query time and conditions the generation on that context.

### Basic RAG Pipeline

1. **Indexing**: documents are chunked, embedded into dense vectors, and stored in a vector database.
2. **Retrieval**: at query time, the question is embedded and the most similar chunks are retrieved via approximate nearest-neighbour (ANN) search.
3. **Generation**: the retrieved chunks are included in the LLM's prompt as context, grounding the answer.

### Advanced RAG Techniques

#### HyDE — Hypothetical Document Embeddings

Instead of embedding the raw query (which may be terse and abstract), HyDE first asks the LLM to generate a hypothetical "ideal answer" and embeds that instead. The hypothetical document is linguistically richer and closer in embedding space to real relevant documents.

#### Multi-Query Retrieval

A single query may miss relevant chunks phrased differently. Multi-query generates N paraphrases of the original question, retrieves candidates for each, and merges the results. This boosts recall significantly.

#### Reciprocal Rank Fusion (RRF)

RRF combines ranked result lists from multiple queries without needing score normalisation. Each document's fused score is `sum(1 / (k + rank_i))` across queries where it appears, where k is a constant (typically 60). RRF consistently outperforms simple score averaging.

#### Cross-Encoder Re-ranking

Bi-encoder models (used for initial retrieval) embed queries and documents independently — they are fast but approximate. Cross-encoders process the (query, document) pair jointly, producing more accurate relevance scores. After initial retrieval, a cross-encoder re-ranks the top-k candidates to surface the most relevant ones.

#### Contextual Compression

Retrieved chunks may contain irrelevant sentences. Contextual compression asks the LLM to extract only the sentences relevant to the query before including them in the prompt. This reduces noise and fits more relevant information within the context window.

## Vector Databases

Vector databases store and index high-dimensional embeddings for fast similarity search. Key systems include:

| Database | Notes |
|----------|-------|
| ChromaDB | Lightweight, embedded, great for local development |
| Pinecone | Managed cloud service, highly scalable |
| Weaviate | Open-source, GraphQL API, supports multi-modal |
| Qdrant | Rust-based, very fast, self-hostable |
| FAISS | Facebook's library, not a full DB but widely used |

### Similarity Metrics

- **Cosine similarity**: measures the angle between vectors; ignores magnitude. Best for semantic embeddings.
- **Dot product**: similar to cosine but magnitude-sensitive; preferred when embeddings are normalised.
- **Euclidean (L2) distance**: measures straight-line distance; suitable for dense embedding spaces.

## Embedding Models

Embedding models convert text (or images) into fixed-size dense vectors that capture semantic meaning.

- **all-MiniLM-L6-v2**: fast, 384-dimensional, great for general semantic search
- **all-mpnet-base-v2**: higher quality, 768-dimensional
- **CLIP (clip-ViT-B-32)**: multimodal model that embeds both text and images in a shared 512-dimensional space, enabling cross-modal retrieval
- **BGE-M3**: state-of-the-art multilingual dense + sparse retrieval

## Multimodal AI

Multimodal AI systems process and relate multiple data types — text, images, audio, video — within a single model or pipeline.

Vision-language models (VLMs) like LLaVA combine a visual encoder (e.g., CLIP) with a language model backbone to answer questions about images, generate captions, and reason visually. CLIP embeddings enable cross-modal retrieval: a text query can retrieve relevant images by comparing their embeddings in a shared space.

In a multimodal RAG system, the knowledge base contains both text chunks and image embeddings. At query time, the system searches both modalities, fuses the results, and presents the LLM with textual context alongside image captions or raw images (for vision-capable models).
