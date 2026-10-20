---
title: TinyStories JAX GPT
emoji: 🐻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
tags:
  - jax
  - flax
  - text-generation
  - tinystories
  - gpt
short_description: ~20M GPT story generator trained with JAX / Flax NNX
---

# TinyStories GPT (JAX / Flax NNX)

Interactive demo for a ~20M-parameter decoder-only language model trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

Built from scratch with **JAX** and **Flax NNX** — see the full training notebook
in the [GitHub repo](https://github.com/JosselinPerret/llm-with-Jax).

## How to use

1. Enter a short story prompt  
2. Adjust temperature (lower = more deterministic)  
3. Choose how many tokens to generate  
4. Click **Submit**

## Model card

| Spec | Value |
| --- | --- |
| Parameters | ~20.2M |
| Context | 128 tokens |
| Embedding | 192 |
| Heads | 6 |
| Blocks | 6 |
| Tokenizer | GPT-2 (`tiktoken`) |
| Hardware | CPU (Space free tier) |
