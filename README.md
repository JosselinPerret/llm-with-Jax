

# SmallGPT

**a ~20M GPT from scratch with JAX & Flax NNX**



[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Space-FFD21E?style=for-the-badge&labelColor=0B0F19)](https://huggingface.co/spaces/almostjojo/tinystories-jax-gpt)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/JosselinPerret/llm-with-Jax)
[![TinyStories](https://img.shields.io/badge/Dataset-TinyStories-blue?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/roneneldan/TinyStories)

![JAX](https://img.shields.io/badge/JAX-0B5CAF?style=flat-square&logo=jax&logoColor=white)![Flax](https://img.shields.io/badge/Flax%20NNX-4B0082?style=flat-square)![Params](https://img.shields.io/badge/Params-~20.2M-success?style=flat-square)![Context](https://img.shields.io/badge/Context-128%20tokens-lightgrey?style=flat-square)![License](https://img.shields.io/badge/License-Learning-orange?style=flat-square)

![llm-with-jax banner](images/banner.png)

---



### Try it now

[![Open demo](https://img.shields.io/badge/Open%20the%20story%20generator-%E2%86%92-black?style=for-the-badge&color=FF4B4B)](https://huggingface.co/spaces/almostjojo/tinystories-jax-gpt)

Type a prompt like *“Once upon a time a big bear…”* and the model continues the story.

---



## What is this?

An end-to-end educational project: a **decoder-only transformer** trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), with:

- custom architecture in **Flax NNX**
- training with **Optax** + **Grain**
- checkpoints via **Orbax**
- text generation + **Gradio** UI
- a public **Hugging Face Space**

---



## Training results

Loss drops fast in the first steps, then settles around ~3 after ~100k stories:

![Training loss curve](images/training_loss.png)

---



## Model card


| Spec               | Value              |
| ------------------ | ------------------ |
| Parameters         | ~20.2M (~81 MB)    |
| Context length     | 128 tokens         |
| Embedding dim      | 192                |
| Attention heads    | 6                  |
| Transformer blocks | 6                  |
| Feed-forward dim   | 512                |
| Tokenizer          | GPT-2 (`tiktoken`) |
| Dataset            | TinyStories        |


**Architecture:** token + positional embeddings → causal multi-head attention → residuals → linear LM head.

---



## Stack

![JAX](https://img.shields.io/badge/JAX-numerics-0B5CAF?style=for-the-badge)![Flax](https://img.shields.io/badge/Flax%20NNX-modules-4B0082?style=for-the-badge)![Optax](https://img.shields.io/badge/Optax-AdamW-1a73e8?style=for-the-badge)![Orbax](https://img.shields.io/badge/Orbax-checkpoints-5c6bc0?style=for-the-badge)![Grain](https://img.shields.io/badge/Grain-dataload-37474f?style=for-the-badge)![Gradio](https://img.shields.io/badge/Gradio-demo-FF7C00?style=for-the-badge)

---



## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

pip install jax flax matplotlib tiktoken grain datasets optax orbax-checkpoint gradio
```

Open `llm-20m-params.ipynb` and run it top to bottom: architecture → data → train → generate → Gradio.

```python
create_story("Once upon a time a big bear ", temperature=0.2, max_new_tokens=30)
```

> Once upon a time a big bear old, who lived in a big forest. He was very happy and he loved to play. One day, he saw a big, a big,

---



## Repo layout

```text
llm-with-Jax/
├── llm-20m-params.ipynb   # full training + inference notebook
├── space/                 # Hugging Face Gradio Space
│   ├── app.py
│   ├── model.py
│   ├── config.json
│   ├── weights.npz
│   └── requirements.txt
├── images/training_loss.png
└── README.md
```

---



## Hugging Face Space

[![Space live](https://img.shields.io/badge/%F0%9F%9A%80%20Space%20is%20live-almostjojo%2Ftinystories--jax--gpt-FFD21E?style=for-the-badge&labelColor=0B0F19)](https://huggingface.co/spaces/almostjojo/tinystories-jax-gpt)

Redeploy from this repo:

```powershell
.\space\deploy.ps1
```

Or:

```powershell
hf repos create almostjojo/tinystories-jax-gpt --type space --space-sdk gradio --flavor zero-a10g --public --exist-ok
hf upload almostjojo/tinystories-jax-gpt .\space . --repo-type space
```

---



## Roadmap

- [x] GPT architecture in Flax NNX  
- [x] TinyStories training pipeline  
- [x] Checkpoints & generation  
- [x] Local Gradio demo  
- [x] Hugging Face Space live  
- [ ] Longer context / larger variants  
- [ ] Faster inference (`jit` + KV cache)

---



## License

For learning and experimentation. Dataset terms: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).