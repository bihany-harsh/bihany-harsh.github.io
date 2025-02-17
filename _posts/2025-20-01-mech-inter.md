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

The following entails my notes for tracking my progress and learning about *mechanistic interpretability*. This blog series would be mix of cut-copy-paste from various sources and where I could, my understanding of the subject as well. I credit this this to [Neel Nanda](https://www.neelnanda.io/about), whose contributions in Anthropic and Deepmind and his awesome blog series and videos over YouTube are my sources.

This particular blog aims to summarise this paper: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). 

# A Mathematical Framework for Transformer Circuits

The aim of **mechinsitic interpretability** is to unveil the black-box that the trasnformer (and a LLM) is.

The paper is majorly concerned with *Attention-only Transformers*. The corresponding structure of the transformer can be interpreted as **sum of vectors (functions) to the *residual stream***. 

> `verbatim` "All components of a transformer (the token embedding, attention heads, MLP layers, and unembedding) communicate with each other by reading and writing to different subspaces of the residual stream. Rather than analyze the residual stream vectors, it can be helpful to decompose the residual stream into all these different communication channels, corresponding to paths through the model." 

As a side note this absolute beauty of a [video](https://youtu.be/9-Jl0dxWQs8?si=aq6-jU-I6RyMeK8y) by Grant Sanderson on his channel [3Blue1Brown](https://www.youtube.com/@3blue1brown) would set a brilliant stage for what in still not understood about the transformer.

## Residual stream and its augmentations

The residual stream is the main *communication channel* for all the layers in a transformer. A key feature is the **linear** as well as **additive** nature of this stream. Every module in the transformer 'reads' from this residual stream and 'writes' back to it. This means between the input and output, the residual stream just gets added onto by a bunch of linear maps. 

> Food for thought...

Hence, given a set of (input, output) pairs, the set of modules contributing most to this change can be observed for a given downstream task ~ *selective modelling*. But that is not trivial to achieve because of the underlying composition on the sum that gets feeded into subsequent layers. Not a clean interpretation.

### Lack of a priveleged basis

Individual dimensions (or axes) of the vector space in which the residual stream lies do not carry intrinsic, predefined meaning and the model is free to learn any arbitrary linear combination of these dimentions.

## Virtual weights and Superposition

We concluded that a composition of these maps makes it messy to alienate a particular pair of set of interactions. However *virtual weights* can act as good representations that can help understand this detail. Every module with a transformer 'reads' in the residual stream by **projecting** it into a representation that it *learns* to realize. Also it needs to project back this representation to residual stream in the same dimensionality as that of the input. Consider any attention head, the `d_model` is projected to a `d_head` and then projected-back (or as Neel would call it **embeds**) it to `d_model` back. Also the MLP layer is a classic setting of the same where the projection is from `d_model` to `d_ff = 4*d_model` and the embedding back is in `d_model`. To understand the interaction between any two set of layers (to put it in a crude way: what does *this* module infer (read) from the output of *that* module), virtual weights are decent representations (\\(W_I^iW_O^j, j \leq i\\)). 

### Superposition

The residual stream has a fixed dimensionality. However, it is these fixed *independent* dimensions (or axes) that are accountable memorizing and projecting information from one module in a network to another. But what makes it interesting is something called the *Johnson-Lindenstrauss Lemma*, which states a given some *flexibility* in independence a potentially exponential number of pseudo-independent directions exist. For a much better and visual understanding I recommend watching the latter part of this [video](https://youtu.be/9-Jl0dxWQs8?si=uAmvayX4dS94FsN-) by Grant Sanderson.

> `verbatim` Perhaps because of this high demand on residual stream bandwidth, we've seen hints that some MLP neurons and attention heads may perform a kind of "memory management" role, clearing residual stream dimensions set by other layers by reading in information and writing out the negative version.

> Food for thought...

Fix a task or a set of inputs. Can it be inferred what part of the residual stream is in action. *Why this might be important?* For say a multi-language generation task, one *could* expect a different part of the residual stream contributing to the `unembed` layer's token prediction. Here the key term is different: because sparsity and superposition make it increasingly difficult to define a distinction.

## Attention Heads and their Independence

The attention heads can be re-thought to be independently additive onto the main stream instead of the standard concatenation point of view whence,

$$W_O^H\begin{bmatrix} r^{h_1} \\ r^{h_2} \\ \dots \end{bmatrix} = \left[ W_O^{h_1}, W_O^{h_2}, \dots \right] \begin{bmatrix} r^{h_1} \\ r^{h_2} \\ \dots \end{bmatrix} = \sum_i W_O^{h_i}r^{h_i}$$

Since time immemorial: **attention heads move information**. An interesting point of view presented in the paper in the separability of information being "read" and being "written". *How could this be the case?*

Consider the attention pattern \\( A = \texttt{softmax}\left( x^T W_Q^T W_K x \right) \\). As long as the product of the \\(Q^TK\\) remains the same the attentions scores remain the same. Attention itself can be written in this form

$$h(x) = (A \otimes W_OW_V) \cdot x = A x W_V^TW_O$$

Notice the embedding back is through a *separate* circuit (not exactly). Kindly forgive my lack of mathematical formalism but the main takeaways are:

- This copying and pasting of information amongst tokens generally happens in a different subspace and the emergence of this back to the residual vector space is the 'contexualised word embeddings'. What attention has achieved is add to static embeddings (at least in the first layer) very context-specific information which it has aggregated and borrowed from other tokens. Thus a static embedding for a word like 'bank' differentiates in the context of finance and rivers.

- > `verbatim` An attention head is really applying two linear operations, \\(A\\) and \\(W_OW_V\\), which operate on differnt dimensions and act independently
  - > \\(A\\) governs which token's information is moved from and to.
  - > \\(W_OW_V\\) governs which information is read from the source token and how it is written to the destination token.

- Non-linearity introduced by \\(A\\) due to the application of \\(\texttt{softmax}\\).

- The \\(QK\\) and \\(OV\\) matrices operate together. They as a group are not independent. The application is parametrized as a low-rank matrices where the \\(OV\\) circuit is described by \\(W_OW_V\\) and similarly the other \\(QK\\) circuit.

- > `verbatim` Product of attention heads behave much like attention heads themselves.

This becomes interesting in the discussion of attention-only transformers.

(T.B.C. soon)

---

You can please choose to ignore the entire write-up and watch this much more informed and articulate summary by Neel Nanda: [video](https://youtu.be/KV5gbOmHbjU?si=IC0O_H31iE9l64Od) 
{: .notice}