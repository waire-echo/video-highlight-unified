# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Scoped inference compatibility for RGBRank trained with Transformers5.5.4.
# Does not replace or monkeypatch the installed Transformers package.
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel as CurrentVisionModel

class Qwen3_5VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class RGBTrainingVisionModel(CurrentVisionModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        head_dim=config.hidden_size//config.num_heads
        self.rotary_pos_emb=Qwen3_5VisionRotaryEmbedding(head_dim//2)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
            merge_size = self.spatial_merge_size
            grid_thw_list = grid_thw.tolist()

            max_hw = max(max(h, w) for _, h, w in grid_thw_list)
            freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
            device = freq_table.device

            total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
            pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

            offset = 0
            for num_frames, height, width in grid_thw_list:
                merged_h, merged_w = height // merge_size, width // merge_size

                block_rows = torch.arange(merged_h, device=device)  # block row indices
                block_cols = torch.arange(merged_w, device=device)  # block col indices
                intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
                intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

                # Compute full-resolution positions
                row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
                col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

                row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
                col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

                coords = torch.stack((row_idx, col_idx), dim=-1)

                if num_frames > 1:
                    coords = coords.repeat(num_frames, 1)

                num_tokens = coords.shape[0]
                pos_ids[offset : offset + num_tokens] = coords
                offset += num_tokens

            embeddings = freq_table[pos_ids]  # lookup rotary embeddings
            embeddings = embeddings.flatten(1)
            return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
            grid_thw_list = grid_thw.tolist()
            grid_ts = [row[0] for row in grid_thw_list]
            grid_hs = [row[1] for row in grid_thw_list]
            grid_ws = [row[2] for row in grid_thw_list]
            device = self.pos_embed.weight.device

            idx_list = [[] for _ in range(4)]
            weight_list = [[] for _ in range(4)]

            for t, h, w in grid_thw_list:
                h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
                w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

                h_idxs_floor = h_idxs.int()
                w_idxs_floor = w_idxs.int()
                h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
                w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

                dh = h_idxs - h_idxs_floor
                dw = w_idxs - w_idxs_floor

                base_h = h_idxs_floor * self.num_grid_per_side
                base_h_ceil = h_idxs_ceil * self.num_grid_per_side

                indices = [
                    (base_h[None].T + w_idxs_floor[None]).flatten(),
                    (base_h[None].T + w_idxs_ceil[None]).flatten(),
                    (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                    (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
                ]

                weights = [
                    ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                    ((1 - dh)[None].T * dw[None]).flatten(),
                    (dh[None].T * (1 - dw)[None]).flatten(),
                    (dh[None].T * dw[None]).flatten(),
                ]

                for i in range(4):
                    idx_list[i].extend(indices[i].tolist())
                    weight_list[i].extend(weights[i].tolist())

            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
            weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
            pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
            patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

            patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

            patch_pos_embeds_permute = []
            merge_size = self.config.spatial_merge_size
            for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
                pos_embed = pos_embed.repeat(t, 1)
                pos_embed = (
                    pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                    .permute(0, 1, 3, 2, 4, 5)
                    .flatten(0, 4)
                )
                patch_pos_embeds_permute.append(pos_embed)
            patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
            return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
            """
            Args:
                hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                    The final hidden states of the model.
                grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                    The temporal, height and width of feature shape of each image in LLM.

            Returns:
                `torch.Tensor`: hidden_states.
            """
            hidden_states = self.patch_embed(hidden_states)

            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
            hidden_states = hidden_states + pos_embeds

            rotary_pos_emb = self.rot_pos_emb(grid_thw)

            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                # Select dtype based on the following factors:
                #  - FA2 requires that cu_seqlens_q must have dtype int32
                #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
                # See https://github.com/huggingface/transformers/pull/34852 for more information
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            for blk in self.blocks:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            merged_hidden_states = self.merger(hidden_states)

            return BaseModelOutputWithPooling(
                last_hidden_state=hidden_states,
                pooler_output=merged_hidden_states,
            )
