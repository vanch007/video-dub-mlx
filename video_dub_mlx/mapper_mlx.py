"""MLX implementation of Hidden2CVMapper for Echo-S2ST."""
import json
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=896, n_head=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj_weight = None  # (3*d_model, d_model)
        self.in_proj_bias = None    # (3*d_model,)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def _self_attn(self, x, mask=None):
        B, T, C = x.shape
        # in_proj: x @ W.T + b
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape to (B, n_head, T, head_dim)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            # mask: (B, 1, 1, T)
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v  # (B, n_head, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)

    def __call__(self, x, mask=None):
        # Pre-LN
        h = self.norm1(x)
        x = x + self._self_attn(h, mask)
        h = self.norm2(x)
        ff = self.linear2(nn.gelu(self.linear1(h)))
        x = x + ff
        return x


class MLXHidden2CVMapper(nn.Module):
    def __init__(self, d_h=4096, d_e=896, n_layer=3, n_head=8, d_ff=2048, max_len=512, n_lang=6):
        super().__init__()
        self.d_e = d_e
        self.n_lang = n_lang
        if n_lang > 0:
            self.lang_emb = nn.Embedding(n_lang, d_e)
        self.h_norm = nn.LayerNorm(d_h)
        self.h_proj = nn.Linear(d_h, d_e)
        self.e_norm = nn.LayerNorm(d_e)
        self.e_proj = nn.Linear(d_e, d_e)
        self.pos = mx.zeros((1, max_len, d_e))

        self.layers = [TransformerEncoderLayer(d_e, n_head, d_ff) for _ in range(n_layer)]
        self.out = nn.Linear(d_e, d_e)

    def __call__(self, h, e, mask=None, lang=None):
        """
        h: (B, m, d_h) float32
        e: (B, m, d_e) float32
        lang: (B,) int32
        """
        m = h.shape[1]
        x = self.h_proj(self.h_norm(h)) + self.e_proj(self.e_norm(e)) + self.pos[:, :m]
        if self.n_lang > 0 and lang is not None:
            x = x + mx.expand_dims(self.lang_emb(lang), 1)
        for layer in self.layers:
            x = layer(x, mask)
        return e + self.out(x)

    @classmethod
    def from_pretrained(cls, config_path, weights_path):
        with open(config_path) as f:
            cfg = json.load(f)
        langs = cfg.pop("langs", None)
        model = cls(
            d_h=cfg.get("d_h", 4096),
            d_e=cfg.get("d_e", 896),
            n_layer=cfg.get("n_layer", 3),
            n_head=cfg.get("n_head", 8),
            d_ff=cfg.get("d_ff", 2048),
            max_len=cfg.get("max_len", 512),
            n_lang=cfg.get("n_lang", 6),
        )
        with safe_open(weights_path, framework="pt") as f:
            sd = {k: f.get_tensor(k).float().numpy() for k in f.keys()}

        model.h_norm.weight = mx.array(sd["h_norm.weight"])
        model.h_norm.bias = mx.array(sd["h_norm.bias"])
        model.h_proj.weight = mx.array(sd["h_proj.weight"])
        model.h_proj.bias = mx.array(sd["h_proj.bias"])

        model.e_norm.weight = mx.array(sd["e_norm.weight"])
        model.e_norm.bias = mx.array(sd["e_norm.bias"])
        model.e_proj.weight = mx.array(sd["e_proj.weight"])
        model.e_proj.bias = mx.array(sd["e_proj.bias"])

        if model.n_lang > 0 and "lang_emb.weight" in sd:
            model.lang_emb.weight = mx.array(sd["lang_emb.weight"])

        model.pos = mx.array(sd["pos"])
        model.out.weight = mx.array(sd["out.weight"])
        model.out.bias = mx.array(sd["out.bias"])

        for i, layer in enumerate(model.layers):
            layer.norm1.weight = mx.array(sd[f"encoder.layers.{i}.norm1.weight"])
            layer.norm1.bias = mx.array(sd[f"encoder.layers.{i}.norm1.bias"])
            layer.in_proj_weight = mx.array(sd[f"encoder.layers.{i}.self_attn.in_proj_weight"])
            layer.in_proj_bias = mx.array(sd[f"encoder.layers.{i}.self_attn.in_proj_bias"])
            layer.out_proj.weight = mx.array(sd[f"encoder.layers.{i}.self_attn.out_proj.weight"])
            layer.out_proj.bias = mx.array(sd[f"encoder.layers.{i}.self_attn.out_proj.bias"])

            layer.norm2.weight = mx.array(sd[f"encoder.layers.{i}.norm2.weight"])
            layer.norm2.bias = mx.array(sd[f"encoder.layers.{i}.norm2.bias"])
            layer.linear1.weight = mx.array(sd[f"encoder.layers.{i}.linear1.weight"])
            layer.linear1.bias = mx.array(sd[f"encoder.layers.{i}.linear1.bias"])
            layer.linear2.weight = mx.array(sd[f"encoder.layers.{i}.linear2.weight"])
            layer.linear2.bias = mx.array(sd[f"encoder.layers.{i}.linear2.bias"])

        return model
