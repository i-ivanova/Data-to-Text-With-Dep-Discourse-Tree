"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder, TransformerPlanDecoder, TransformerMaskDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder,
           "plan_transformer": TransformerPlanDecoder,
           "mask_transformer": TransformerMaskDecoder
          }

__all__ = ["DecoderBase", "TransformerDecoder", "TransformerPlanDecoder", "TransformerMaskDecoder",
           "StdRNNDecoder", "CNNDecoder", "InputFeedRNNDecoder", "str2dec"]
