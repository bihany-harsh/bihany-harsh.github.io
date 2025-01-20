---
title: My Notes for Mechanistic Interpretability
date: 2025-01-19
permalink: /posts/2025/01/mech_interpret/
tags:
  - LLMs
  - Language Processing
  - Algorithms
  - Transformers
---

# Mechanistic Interepretability (Part 1)

The following entails my notes for tracking my progress and learning about *mechanistic interpretability*. This blog series would be mix of cut-copy-paste from various sources and when I could my understanding of the subject. I credit this this to [Neel Nanda](https://www.neelnanda.io/about), whose contributions in Anthropic and Deepmind and his awesome blog series and videos over YouTube are my sources.

This particular blog aims to summarise this paper: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). 

# A Mathematical Framework for Transformer Circuits

The aim of **mechinsitic interpretability** is to unveil the black-box that the trasnformer (and an LLM) is.

The paper is majorly concerned with *Attention-only Transformers*. The corresponding structure of the transformer can be interpreted as **sum of vectors (functions) to the *residual stream***. 

> `verbatim` "All components of a transformer (the token embedding, attention heads, MLP layers, and unembedding) communicate with each other by reading and writing to different subspaces of the residual stream. Rather than analyze the residual stream vectors, it can be helpful to decompose the residual stream into all these different communication channels, corresponding to paths through the model." 

As a side note this absolute beauty of a [video](https://youtu.be/9-Jl0dxWQs8?si=aq6-jU-I6RyMeK8y) by Grant Sanderson on his channel [3Blue1Brown](https://www.youtube.com/@3blue1brown) would set a brilliant stage for what in still not understood about the transformer.



