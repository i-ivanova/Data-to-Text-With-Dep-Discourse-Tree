#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import torch

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    graph_shards = split_corpus(opt.graph, opt.shard_size)
    plan_shard = split_corpus(opt.plan, opt.shard_size)
    tree_shard = split_corpus(opt.tree, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    shard_pairs = zip(src_shards, graph_shards, tgt_shards)

    for i, (src_shard, graph_shard, plan_shard, tree_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            graph=graph_shard,
            plan=plan_shard,
            tree=tree_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def beam_search(model, start_token, end_token, k, max_len=100):
    """
    Beam search implementation for language model decoding.
    :param model: the model to use for decoding
    :param start_token: the start token
    :param end_token: the end token
    :param k: the number of beams to use
    :param max_len: the maximum length of the sequence to generate
    :return: the list of decoded sequences
    """
    model.eval()
    # Initialize the starting input
    k_sequences = [[[start_token], 0.0]]
    # Iterate until we have k sequences of length max_len
    for _ in range(max_len):
        # Initialize the list of new sequences
        new_sequences = []
        # Iterate over the current sequences
        for sequence, score in k_sequences:
            # Get the last token of the sequence
            last_token = sequence[-1]
            # Get the logits for the next token
            logits, _ = model(sequence)
            # Get the top k logits
            top_k_logits = torch.topk(logits[0, -1, :], k)[0]
            # Iterate over the top k logits
            for logit in top_k_logits:
                # Get the new token
                new_token = torch.multinomial(torch.softmax(logit, dim=0), 1)[0]
                # Get the new score
                new_score = score + logit.item()
                # Add the new sequence to the list of new sequences
                new_sequences.append([sequence + [new_token], new_score])
        # Sort the new sequences by score
        new_sequences.sort(key=lambda x: x[1], reverse=True)
        # Get the best k sequences
        k_sequences = new_sequences[:k]
    # Return the best k sequences
    return k_sequences


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
