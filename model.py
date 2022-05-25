# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang, Zenwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from torch.profiler import record_function


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and
            (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output
            contains unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

        self.simple_am_proj = ScaledLinear(encoder_dim, vocab_size, initial_speed=0.5)
        self.simple_lm_proj = ScaledLinear(decoder_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.25,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          targets:
            A 2-D tensor of shape (N, U). It contains the labels for utterances.
          target_lengths:
            A 1-D tensor of shape (N,). It contains the number of tokens in `targets`.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.size(0) == x_lens.size(0) == targets.size(0) == target_lengths.size(0)
        blank_id = self.decoder.blank_id

        with record_function("rnnt_encoder"):
            encoder_out, encoder_out_lengths = self.encoder(x, x_lens)
        assert torch.all(encoder_out_lengths > 0)

        # [B, 1 + S]
        decoder_in = nn.functional.pad(targets, (1, 0))
        with record_function("rnnt_decoder"):
            # decoder_out: [B, S + 1, decoder_dim]
            decoder_out = self.decoder(decoder_in)

        begin = torch.zeros_like(target_lengths)
        boundary = torch.stack(
            [begin, begin, target_lengths, encoder_out_lengths], dim=1
        )

        with record_function("rnnt_loss_simple"):
            lm = self.simple_lm_proj(decoder_out)
            am = self.simple_am_proj(encoder_out)
            with torch.cuda.amp.autocast(enabled=False):
                simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                    lm=lm.float(),
                    am=am.float(),
                    symbols=targets,
                    termination_symbol=blank_id,
                    lm_only_scale=lm_scale,
                    am_only_scale=am_scale,
                    boundary=boundary,
                    reduction="sum",
                    return_grad=True,
                )

        with record_function("rnnt_loss_pruned"):
            # ranges : [B, T, prune_range]
            ranges = k2.get_rnnt_prune_ranges(
                px_grad=px_grad,
                py_grad=py_grad,
                boundary=boundary,
                s_range=prune_range,
            )
            # am_pruned : [B, T, prune_range, encoder_dim]
            # lm_pruned : [B, T, prune_range, decoder_dim]
            am_pruned, lm_pruned = k2.do_rnnt_pruning(
                am=self.joiner.encoder_proj(encoder_out),
                lm=self.joiner.decoder_proj(decoder_out),
                ranges=ranges,
            )
            # logits : [B, T, prune_range, vocab_size]
            # project_input=False since we applied the decoder's input projections
            # prior to do_rnnt_pruning (this is an optimization for speed).
            logits = self.joiner(am_pruned, lm_pruned, project_input=False)
            with torch.cuda.amp.autocast(enabled=False):
                pruned_loss = k2.rnnt_loss_pruned(
                    logits=logits.float(),
                    symbols=targets,
                    ranges=ranges,
                    termination_symbol=blank_id,
                    boundary=boundary,
                    reduction="sum",
                )

        return (simple_loss, pruned_loss)
