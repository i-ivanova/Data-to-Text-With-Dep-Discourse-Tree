# -*- coding: utf-8 -*-
from functools import partial

import six
import torch
from torchtext.data import Field, RawField

from onmt.inputters.datareader_base import DataReaderBase
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

from collections import Counter

class TextDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


class TextMultiField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data

        feats = [ff.process(batch_by_feat[i], device=device)
                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """

        return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def text_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim)
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field


def node_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            feat_delim=feat_delim)
        use_len = i == 0 and include_lengths
        feat = Field(
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field


class GraphField(RawField):

    def preprocess(self, x):

        edge_index_1 = []
        edge_index_2 = []
        edges_types = []

        x = x.strip().split()
        for e in x:
            e = e.replace('(', '').replace(')', '')
            e = e.split(',')
            e2, e1, l, ln = e

            edge_index_1.append(int(e1))
            edge_index_2.append(int(e2))
            edges_types.append(int(l))

        edges_index = torch.tensor([edge_index_2, edge_index_1], dtype=torch.long)
        edges_types = torch.tensor(edges_types, dtype=torch.long)

        return edges_index, edges_types

class PlanField(RawField):
    
    def __init__(self, field_kwargs):
        super().__init__(field_kwargs)
        self.num_vocab = 4
        self.vocab = Vocab(Counter({"<edu_end>": 1}))
        
    def preprocess(self, x):
        
        plan_seq = []
        x = x.strip().split()
        plan_seq.append(-2) # BOS
        for elm in x:
            if elm == "<edu_end>":
                plan_seq.append(-1) # <edu_end>
            else:
                plan_seq.append(int(elm))
        plan_seq.append(-1)
        plan_seq.append(-3) # EOS
        plan_seq = torch.tensor(plan_seq, dtype=torch.long)
        return plan_seq
    
    def process(self, x_list, device):
        lengths = [len(x) for x in x_list]
        batch = pad_sequence(x_list, batch_first=True, padding_value=-4).to(device) # PAD
        lengths = torch.tensor(lengths).to(device)
        return batch, lengths

fine_to_coarse_map = {
    
    'ROOT': 'root',
    'attribution': 'attribution',
    'Related': 'background',
    'bg-goal': 'background',
    'bg-general': 'background',
    'cause': 'cause-effect',
    'condition': 'condition',
    'result': 'cause-effect',
    'bg-compare': 'background',
    'comparison': 'comparison',
    'contrast': 'contrast',
    'elab-addition': 'elaboration',
    'elab-aspect': 'elaboration',
    'elab-process_step': 'elaboration',
    'elab-definition': 'elaboration',
    'elab-enum_member': 'elaboration',
    'elab-example': 'elaboration',
    'enablement': 'enablement',
    'evaluation': 'evaluation',
    'exp-evidence': 'explain',
    'exp-reason': 'explain',
    'joint': 'joint',
    'manner-means': 'manner-means',
    'progression': 'progression',
    'same-unit': 'same-unit',
    'summary': 'summary',
    'temporal': 'temporal',
    'null': 'null'
}


rel2idx = {}
rel_id = 0
for rel in fine_to_coarse_map.values():
    if rel not in rel2idx:
        rel2idx[rel] = rel_id
        rel_id += 1


class TreeField(RawField):
    
    def __init__(self, field_kwargs):
        super().__init__(field_kwargs)
    
    def preprocess(self, x):
        
        parents = []
        rels = []
        x = x.strip().split()
        for i in range(0, len(x), 2):
            parents.append(int(x[i]))
            rels.append(rel2idx[x[i+1]])

        parents = torch.tensor(parents, dtype=torch.long)
        rels = torch.tensor(rels, dtype=torch.long)

        return parents, rels
