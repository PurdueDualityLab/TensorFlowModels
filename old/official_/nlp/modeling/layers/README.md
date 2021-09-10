# Layers

Layers are the fundamental building blocks for NLP models. They can be used to
assemble new layers, networks, or models.

*   [MultiHeadAttention](attention.py) implements an optionally masked attention
    between query, key, value tensors as described in
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). If
    `from_tensor` and `to_tensor` are the same, then this is self-attention.

*   [CachedAttention](attention.py) implements an attention layer with cache
    used for auto-agressive decoding.

*   [MatMulWithMargin](mat_mul_with_margin.py) implements a matrix
    multiplication with margin layer used for training retrieval / ranking
    tasks, as described in ["Improving Multilingual Sentence Embedding using
    Bi-directional Dual Encoder with Additive Margin
    Softmax"](https://www.ijcai.org/Proceedings/2019/0746.pdf).

*   [MultiChannelAttention](multi_channel_attention.py) implements an variant of
    multi-head attention which can be used to merge multiple streams for
    cross-attentions.

*   [TalkingHeadsAttention](talking_heads_attention.py) implements the talking
    heads attention, as decribed in
    ["Talking-Heads Attention"](https://arxiv.org/abs/2003.02436).

*   [Transformer](transformer.py) implements an optionally masked transformer as
    described in
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

*   [TransformerDecoderBlock](transformer.py) TransformerDecoderBlock is made up
    of self multi-head attention, cross multi-head attention and feedforward
    network.

*   [ReZeroTransformer](rezero_transformer.py) implements Transformer with
    ReZero described in
    ["ReZero is All You Need: Fast Convergence at Large Depth"](https://arxiv.org/abs/2003.04887).

*   [OnDeviceEmbedding](on_device_embedding.py) implements efficient embedding
    lookups designed for TPU-based models.

*   [PositionalEmbedding](position_embedding.py) creates a positional embedding
    as described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805).

*   [SelfAttentionMask](self_attention_mask.py) creates a 3D attention mask from
    a 2D tensor mask.

*   [MaskedSoftmax](masked_softmax.py) implements a softmax with an optional
    masking input. If no mask is provided to this layer, it performs a standard
    softmax; however, if a mask tensor is applied (which should be 1 in
    positions where the data should be allowed through, and 0 where the data
    should be masked), the output will have masked positions set to
    approximately zero.

*   [`MaskedLM`](masked_lm.py) implements a masked language model. It assumes
    the embedding table variable is passed to it.

*   [ClassificationHead](cls_head.py) A pooling head over a sequence of
    embeddings, commonly used by classification tasks.

*   [GatedFeedforward](gated_feedforward.py) implements the gated linear layer
    feedforward as described in
    ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202).

*   [MultiHeadRelativeAttention](relative_attention.py) implements a variant
    of multi-head attention with support for relative position encodings as
    described in "Transformer-XL: Attentive Language Models Beyond a
    Fixed-Length Context"(https://arxiv.org/abs/1901.02860). This also has
    extended support for segment-based attention, a re-parameterization
    introduced in "XLNet: Generalized Autoregressive Pretraining for Language
    Understanding" (https://arxiv.org/abs/1906.08237).

*   [TwoStreamRelativeAttention](relative_attention.py) implements a variant
    of multi-head relative attention as described in "XLNet: Generalized
    Autoregressive Pretraining for Language Understanding"
    (https://arxiv.org/abs/1906.08237). This takes in a query and content
    stream and applies self attention.

*   [TransformerXL](transformer_xl.py) implements Transformer XL introduced in
    "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    (https://arxiv.org/abs/1901.02860). This contains `TransformerXLBlock`, a
    block containing either one or two stream relative self-attention as well as
    subsequent feedforward networks. It also contains `TransformerXL`, which
    contains attention biases as well as multiple `TransformerXLBlocks`.

*   [MobileBertEmbedding](mobile_bert_layers.py) and
    [MobileBertTransformer](mobile_bert_layers.py) implement the embedding layer
    and also transformer layer proposed in the
    [MobileBERT paper](https://arxiv.org/pdf/2004.02984.pdf).

*   [BertPackInputs](text_layers.py) and
    [BertTokenizer](text_layers.py) and [SentencepieceTokenizer](text_layers.py)
    implements the layer to tokenize raw text and pack them into the inputs for
    BERT models.
