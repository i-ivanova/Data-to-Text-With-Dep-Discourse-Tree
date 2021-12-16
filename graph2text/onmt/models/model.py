""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
import dgl


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, batch=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs
        enc_state, memory_bank, lengths = self.encoder(src, lengths, batch=batch)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

class NMTPlanModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
      plan_decoder (onmt.decoders.DecoderBase): a decoder object for the EDU-entity plan
    """

    def __init__(self, encoder, decoder, plan_decoder, tree_decoder):
        super(NMTPlanModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.plan_decoder = plan_decoder
        self.tree_decoder = tree_decoder

    def _fix_corrupted_plans(self, plan, tree_sizes):
        plan_size = (plan == -1).sum()
        tree_nodes = tree_sizes.sum()
        if (plan_size != tree_nodes):
            print("CORRPUTED TREE IN PASS")
            for i in range(tree_sizes.shape[0]):
                # corrupted plan because of empty edu start
                if tree_sizes[i] != (plan[i] == -1).sum():
                    plan[i] = torch.cat((plan[i][:1], plan[i][2:], torch.tensor([-4],
                    device=torch.cuda.current_device()).long()))
        return plan
    

    def forward(self, src, plan, tgt, trees, src_lengths, plan_lenths, bptt=False, with_align=False, batch=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            src_lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.
        Returns:
            (FloatTensor, dict[str, FloatTensor], FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
            * plan decoder output ``(plan_len, batch, hidden)``
            * plan dictionary attention dists of ``(plan_len, batch, src_len)``
        """
        # print("TODO Why exclude last target from inputs, NMTPlanModel")
        dec_in = tgt#[:-1]  # exclude last target from inputs 
        enc_state, memory_bank, src_lengths = self.encoder(src, src_lengths, batch=batch)
        
        # print(enc_state.shape, memory_bank.shape, src.shape, tgt.shape, plan.shape, with_align)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
            self.plan_decoder.init_state(src, memory_bank, enc_state)

        tree_sizes = None
        if self.tree_decoder:
            tree_sizes = torch.tensor([tree_inst[0].shape for tree_inst in trees],
                                     device=torch.cuda.current_device())
            plan = self._fix_corrupted_plans(plan, tree_sizes)


        plan_dec_out, plan_attns = self.plan_decoder(plan, memory_bank,
                                      memory_lengths=src_lengths,
                                      with_align=with_align)
        # print("Generating Blan: ")
        # print("DECODER IN ", dec_in)
        # print("ENC STATE: ", enc_state); print("SHAPE ", enc_state.shape)
        # print("Memory Bank: ", memory_bank.shape, memory_bank)
        # print("PLAN", plan.shape, plan)
        # print("INPUTS ", plan_dec_out.shape, plan_attns["std"].shape)
        # self.plan_generator(plan_dec_out, plan_attns, None)
        # raise ValueError
        
        tree_loss = 0
        num_trees = len(trees)
        if self.tree_decoder:
            device = torch.cuda.current_device()
            plan_batch, plan_length = plan.shape
            # num_trees = len(trees)
            
            #print("TREE ", trees[0])

            edu_idx = (plan == -1).view(plan_length, plan_batch)
            #print("PLAN ", plan)
            # tree_sizes = torch.tensor([tree_inst[0].shape for tree_inst in trees],
            #                          device=torch.cuda.current_device())
            
            #print("TREE SIZES ", tree_sizes)
            #print("PLAN_DEC ", plan_dec_out[edu_idx].shape, edu_idx.shape, plan_dec_out[edu_idx])
            tree_edus = torch.split(plan_dec_out[edu_idx], tuple(tree_sizes.view(-1)))
            
            for i, tree in enumerate(trees):
                arcs, rels = tree
                #print(arcs)
                num_nodes = len(arcs)
                # TODO: stop omiting trees with only root node
                if num_nodes < 2:
                    continue
                    
                left_adj, right_adj = torch.zeros((1, num_nodes, num_nodes, 1), device=device), \
                                        torch.zeros((1, num_nodes, num_nodes, 1), device=device)
                root = (arcs == -1).nonzero(as_tuple=True)[0]
                edges = []
                for j in range(num_nodes):
                    node_id, node_parent = j, arcs[j]
                    if node_parent == -1:
                        root = node_id
                    else:
                        if j < node_parent:
                            left_adj[0, node_parent, node_id, 0] = 1
                        else:
                            right_adj[0, node_parent, node_id, 0] = 1
                        edges.append((node_parent, node_id))
                
                tree_graph = dgl.DGLGraph(edges).to("cuda:0")
                edus = tree_edus[i].unsqueeze(0)
                adj_matrix = (left_adj, right_adj)
                # tree_loss += self.tree_decoder(edus, tree_graph, adj_matrix, root, rels)
                compat_matrix_full = self.tree_decoder.get_compat_matrix(edus.squeeze(0))
                root_scores = self.tree_decoder.root_clf(edus).view(-1)
                # Decode the tree structure
                msp_result, etype, pred_root = self.tree_decoder.decode_mst(compat_matrix_full, root_scores)
                # Decode the EDU order from the tree
                dep_tree_root, new_adj_matrix = self.tree_decoder.arrange_dep_tree_rootclf(msp_result, etype, int(pred_root))
                # print("Dep tree root", dep_tree_root)
                flat_new_adj = torch.sum(new_adj_matrix, dim=2)
                adj = torch.cat([adj_matrix[0].squeeze(0), adj_matrix[1].squeeze(0)], dim=2)
                flat_gold_adj = torch.sum(adj, dim=2)               
                uas = 1 - torch.sum(flat_new_adj != flat_gold_adj).float() / (num_nodes * 2) - int(pred_root != root) / num_nodes
                las = 1 - torch.sum(new_adj_matrix != adj).float() / (num_nodes * 2) - int(pred_root != root) / num_nodes                
                print("UAS: ", uas.item())
                print("LAS: ", las.item())
                
                pred_order = self.tree_decoder.node_order((edus, torch.mean(edus, dim=1)), dep_tree_root)
                # Make 1 to num_nodes instead of 0 to num_nodes - 1
                pred_order = [i + 1 for i in pred_order]
                print("Order: ", pred_order)

        
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=src_lengths,
                                      with_align=with_align)
        tree_loss /= num_trees
        return dec_out, plan_dec_out, attns, plan_attns, tree_loss

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
