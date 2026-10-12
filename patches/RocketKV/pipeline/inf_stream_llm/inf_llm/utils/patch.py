# MIT License

# Copyright (c) 2024 jiayi yuan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA License

# =======================================================================

# 1. Definitions

# “Licensor” means any person or entity that distributes its Work.

# “Work” means (a) the original work of authorship made available under
# this license, which may include software, documentation, or other files,
# and (b) any additions to or derivative works thereof that are made
# available under this license.

# The terms “reproduce,” “reproduction,” “derivative works,” and “distribution”
# have the meaning as provided under U.S. copyright law; provided, however,
# that for the purposes of this license, derivative works shall not include works
# that remain separable from, or merely link (or bind by name) to the
# interfaces of, the Work.

# Works are “made available” under this license by including in or with the Work
# either (a) a copyright notice referencing the applicability of
# this license to the Work, or (b) a copy of this license.

# 2. License Grant

# 2.1 Copyright Grant. Subject to the terms and conditions of this license, each
# Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free,
# copyright license to use, reproduce, prepare derivative works of, publicly display,
# publicly perform, sublicense and distribute its Work and any resulting derivative
# works in any form.

# 3. Limitations

# 3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under
# this license, (b) you include a complete copy of this license with your distribution,
# and (c) you retain without modification any copyright, patent, trademark, or
# attribution notices that are present in the Work.

# 3.2 Derivative Works. You may specify that additional or different terms apply to the use,
# reproduction, and distribution of your derivative works of the Work (“Your Terms”) only
# if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative
# works, and (b) you identify the specific derivative works that are subject to Your Terms.
# Notwithstanding Your Terms, this license (including the redistribution requirements in
# Section 3.1) will continue to apply to the Work itself.

# 3.3 Use Limitation. The Work and any derivative works thereof only may be used or
# intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation
# and its affiliates may use the Work and any derivative works commercially.
# As used herein, “non-commercially” means for research or evaluation purposes only.

# 3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor
# (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that
# you allege are infringed by any Work, then your rights under this license from
# such Licensor (including the grant in Section 2.1) will terminate immediately.

# 3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its
# affiliates’ names, logos, or trademarks, except as necessary to reproduce
# the notices described in this license.

# 3.6 Termination. If you violate any term of this license, then your rights under
# this license (including the grant in Section 2.1) will terminate immediately.

# 4. Disclaimer of Warranty.

# THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT.
# YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE.

# 5. Limitation of Liability.

# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY,
# WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR
# BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR
# INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS
# INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY
# OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGES.

# =======================================================================

import torch
from ..attention import RotaryEmbeddingESM, ATTN_FORWRAD

# transformers >= 4.57 uses cache_position/position_embeddings-era
# decoder interfaces (DynamicCache inside Model.forward, layers called with a cache
# POSITION kwarg, attention returning a 2-tuple). Qwen3 requires >=4.51 anyway. The
# add-on below adapts ONLY the plumbing (Model.forward loop, attention return shape);
# RocketKV's attention/eviction math (rocket*.py) is untouched. On the legacy 4.45.2
# venv every branch below resolves exactly as before (bit-identical behavior).
import transformers as _tf

_tf_major_minor = tuple(int(p) for p in _tf.__version__.split(".")[:2])
TF_457 = _tf_major_minor >= (4, 57)

if TF_457:
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen3ForCausalLM
    from ..attention.rocket_qwen3 import rocket_qwen3_forward
else:
    from transformers import LlamaForCausalLM, MistralForCausalLM

def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions
        if TF_457:
            # 4.57 decoder layers call attention with `past_key_values=` (renamed) and
            # expect a 2-tuple (output, attn_weights). The weight tensor is never used
            # (assert above), and the cache tuple flows back through the 2nd element.
            past_key_value = kwargs.get("past_key_values", past_key_value)
            # 4.57 dropped the num_heads/num_key_value_heads module attributes.
            num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
            num_kv = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        else:
            num_heads = self.num_heads
            num_kv = self.num_key_value_heads
        ret = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_value,
            self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.head_dim, num_heads, num_kv
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None

        if TF_457:
            return o, pkv
        return o, None, pkv

    return hf_forward


def patch_hf(
    model,
    attn_type: str = "inf_llm",
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    rope_theta_factor = None,
    rope_linear_scaling_factor = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM # Qwen2ForCausalLM
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, BaseModelOutputWithPast
    from transformers.models.mistral.modeling_mistral import MistralAttention, MistralModel
    # from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        *args,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        if use_cache:
            pkv = tuple()

        else:
            pkv = None
            

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if TF_457:
            # 4.57-era: patch.py calls the sub-blocks directly so the modern
            # decoder-layer signature (cache_position/position_embeddings/DynamicCache)
            # is bypassed entirely; math and cache layout are identical to the legacy
            # loop below (RocketKV-style per-layer tuple cache inside each attention).
            pkv = tuple()
            for i, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                residual = hidden_states
                hidden_states = decoder_layer.input_layernorm(hidden_states)
                hidden_states, _pkv = decoder_layer.self_attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=self.position_bias,
                    past_key_values=past_key_values[i] if past_key_values is not None else None,
                    use_cache=use_cache,
                )
                hidden_states = residual + hidden_states
                if use_cache:
                    pkv = pkv + (_pkv,)
                # pre-norm residual chain (HF 4.57 decoder layer): the second residual is the
                # POST-attention sum, i.e. y = x + attn(n1(x)); out = y + mlp(n2(y)).
                residual = hidden_states
                hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
                hidden_states = decoder_layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=pkv,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                _cache = layer_outputs[2 if output_attentions else 1]
                pkv = pkv + (_cache,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    if TF_457 and isinstance(model, Qwen3ForCausalLM):
        # Qwen3: QK-norm must be applied before RoPE -> the verbatim rocket.py copy
        # rocket_qwen3.py (QK-norm-aware). Only the "rocket" runner path is used here.
        assert attn_type == "rocket", "Qwen3 add-on supports only attn_type='rocket'"
        _attn_fn = rocket_qwen3_forward
    else:
        _attn_fn = ATTN_FORWRAD[attn_type]
    forward = huggingface_forward(_attn_fn(**attn_kwargs))

    if isinstance(model, LlamaForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, MistralForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, Qwen3ForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif model.__class__.__name__ == "MiniCPMForCausalLM":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    else:
        raise ValueError("Only supports llama, mistral and qwen2 models.")
    
    if TF_457:
        hf_rope = model.model.rotary_emb  # 4.57 moved rotary_emb to Model level
    else:
        hf_rope = model.model.layers[0].self_attn.rotary_emb
    if isinstance(model, LlamaForCausalLM):
        base = base * rope_theta_factor if (base is not None and rope_theta_factor is not None) else hf_rope.config.rope_theta
        distance_scale = distance_scale if distance_scale is not None else 1.0
        rope = RotaryEmbeddingESM(
            hf_rope.config.hidden_size//hf_rope.config.num_attention_heads,
            base,
            distance_scale,
            rope_linear_scaling_factor,
            hf_rope.config
        )
    elif isinstance(model, Qwen3ForCausalLM):
        # Qwen3: vanilla rotary (theta=1e6, no llama3 scaling, no linear scaling) —
        # take head_dim straight from config (Qwen3 varies head_dim independently of
        # hidden_size//num_attention_heads, e.g. 0.6B: 128 vs 1024/16=64).
        base = base * rope_theta_factor if (base is not None and rope_theta_factor is not None) else hf_rope.config.rope_theta
        distance_scale = distance_scale if distance_scale is not None else 1.0
        rope = RotaryEmbeddingESM(
            getattr(hf_rope.config, "head_dim", hf_rope.config.hidden_size//hf_rope.config.num_attention_heads),
            base,
            distance_scale,
            rope_linear_scaling_factor,
            hf_rope.config
        )
    else:
        base = base * rope_theta_factor if (base is not None and rope_theta_factor is not None) else hf_rope.base
        distance_scale = distance_scale if distance_scale is not None else 1.0
        rope = RotaryEmbeddingESM(
            hf_rope.dim,
            base,
            distance_scale,
            rope_linear_scaling_factor
        )
    
    model.model.position_bias = rope
    
    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    model.model._old_forward = model.model.forward
    model.model.forward = model_forward.__get__(model.model, Model)

    return model

