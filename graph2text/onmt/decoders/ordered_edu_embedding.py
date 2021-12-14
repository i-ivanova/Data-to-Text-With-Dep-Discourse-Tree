import torch
import torch.nn as nn
from transformers import BertModel
from dataset.utils.other import pad_and_stack, unpad_toks
from dataset.utils.constants import *

class OrderedEduEmbedding(nn.Module):
    def __init__(self, config):
        super(OrderedEduEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained("SpanBERT/spanbert-base-cased")
        self.linear_out = nn.Linear(1536, config[LSTM_HID])
        self.dropout = nn.Dropout(config[DROPOUT])
        
    def forward(self, tree): 

        # Batching stuff
        all_segs, all_masks, all_seg_lens, per_batch_segs = [], [], [], []
        # Lengths per segment
        seg_lens = [len(s) for s in tree.segments]
        # Get initial BERT embeddings for each token in the document
        # Pad token ids to length of 384 with [PAD] tokens
        segments = pad_and_stack([seg.unsqueeze(1) 
                                  for seg in tree.segments], pad_size=384).squeeze(2).to("cuda:0")
        mask = segments > 0

        # Save for batching
        all_segs.append(segments)
        all_masks.append(mask)
        # Process segments for all batches in parallel
        all_segs, all_masks = torch.cat(all_segs),  \
                                torch.cat(all_masks)
        # Contextualize the embeddings
        context_segments = self.bert(all_segs, 
                                     attention_mask=all_masks.byte())[0]

        # Get linearized token sequence for each document without [PAD]s
        all_context_segments = unpad_toks(context_segments, mask)
        
        edu_embeds = []
        for start, end in tree.sent2subtok_bdry:
            edu_embeds.append(torch.cat([all_context_segments[start], all_context_segments[end]]).unsqueeze(0))
        batch_span_embeds = torch.cat(edu_embeds)
        return self.linear_out(self.dropout(batch_span_embeds))