#!/usr/bin/env python3
#
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
#
# See ../LICENSE for clarification regarding multiple authors
#
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

import argparse

import torch
from torch import nn
from torch.profiler import ProfilerActivity, record_function

from conformer import Conformer
from decoder import Decoder
from joiner import Joiner
from model import Transducer
from utils import (
    AttributeDict,
    ShapeGenerator,
    SortedShapeGenerator,
    generate_data,
    str2bool,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sort-utterance",
        type=str2bool,
        help="True to sort utterance duration before batching them up",
        default=True,
    )

    return parser.parse_args()


def get_model_params() -> AttributeDict:
    """Return a dict containing model parameters."""
    params = AttributeDict(
        {
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "encoder_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            # parameters for decoder
            "decoder_dim": 512,
            "context_size": 2,
            "vocab_size": 500,
            "blank_id": 0,
            # parameters for joiner
            "joiner_dim": 512,
        }
    )
    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    # TODO: We can add an option to switch between Conformer and Transformer
    encoder = Conformer(
        num_features=params.feature_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.encoder_dim,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return model


def main():
    args = get_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(f"device: {device}")

    if args.sort_utterance:
        max_frames = 30000
        suffix = f"max-frames-{max_frames}"
    else:
        # won't OOM when it's 50. Set it to 30 as torchaudio is using 30
        batch_size = 30
        suffix = batch_size

    params = get_model_params()
    print("About to create model")
    model = get_transducer_model(params)
    model.to(device)

    if args.sort_utterance:
        shape_generator = SortedShapeGenerator(max_frames)
    else:
        shape_generator = ShapeGenerator(batch_size)

    print(f"Model profiling started (Sort utterance {args.sort_utterance})")

    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=20, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./log/conformer-k2-pruned-{suffix}"
        ),
        record_shapes=False,
        with_stack=False,
        profile_memory=True,
    )

    prof.start()

    for i, shape_info in enumerate(shape_generator):
        print("i", i)
        feature, feature_lens, targets, target_lengths = generate_data(
            shape_info,
            vocab_size=params.vocab_size,
            num_features=params.feature_dim,
            device=device,
        )
        feature_lens = feature_lens.to(torch.int64)
        target_lengths = target_lengths.to(torch.int64)
        targets = targets.to(torch.int64)

        simple_loss, pruned_loss = model(
            x=feature,
            x_lens=feature_lens,
            targets=targets,
            target_lengths=target_lengths,
        )
        loss = 0.5 * simple_loss + pruned_loss
        assert loss.requires_grad is True

        with record_function("loss-backward"):
            loss.backward()

        model.zero_grad()

        if i > 80:
            break

        prof.step()
    prof.stop()
    print("Profiling done")

    s = str(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=-1,
        )
    )

    with open(f"conformer-k2-pruned-{suffix}.txt", "w") as f:
        f.write(s + "\n")


if __name__ == "__main__":
    torch.manual_seed(20220524)
    main()
