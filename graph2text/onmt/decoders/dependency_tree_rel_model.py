import torch as th
import torch.nn as nn
import dgl
import numpy as np

from onmt.decoders.modules.pointer_net import PointerNet
from onmt.decoders.modules.mst import *
from onmt.utils.span import DepSpanNode
from onmt.utils.constants import *
from collections import defaultdict
import copy
import gc

class DependencyTreeRelModel(nn.Module):
    
    def __init__(self, opt):
        
        super(DependencyTreeRelModel, self).__init__()
        self.device = "cuda"
        self.hid_dim = opt.tree_hid_him
        self.num_lstm_pointer = 1
        self.opt = opt
        
        self.ptr_criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        
        self.bilin_arc = nn.Bilinear(self.hid_dim, self.hid_dim, 2)
        self.head_lin_arc = nn.Linear(self.hid_dim, 2, bias=False)
        self.dep_lin_arc = nn.Linear(self.hid_dim, 2, bias=False)
        
        self.link_predictor = self.bilin_arc_score        
        
        self.root_clf = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim),
                nn.GELU(),
                nn.Linear(self.hid_dim, 1)
        )
                
        self.rel_clf = nn.Sequential(
                nn.Linear(self.hid_dim*2, self.hid_dim),
                nn.GELU(),
                nn.Linear(self.hid_dim, 18)
        )

        
        self.pointer_net = PointerNet(self.hid_dim,
                                      self.hid_dim,
                                      self.num_lstm_pointer,
                                      self.opt.tree_dropout,
                                      self.opt)
        self.alpha = 0.25
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def message_func(self, edges):
        
        return {'src': edges.src['h'], 
                'dst': edges.dst['h'], 
                'ch_h': edges.src['ch_h'],
                'child_ids': edges.edges()[0], 
                'par_ids': edges.edges()[1]}        
        
    def reduce_func(self, nodes, doc_embed, batch_size, seq_len):
        
        num_seqs, num_children, _ = nodes.mailbox['dst'].shape
        if num_children > 1:
            doc_embed_idx = nodes.mailbox['par_ids'][:, 0] // seq_len
            out_prob = self.pointer_net(nodes.mailbox['src'], doc_embed[doc_embed_idx])
            out_prob = out_prob.reshape(-1, out_prob.shape[-1])
            pointer_targets = th.arange(num_children, device=self.config[DEVICE]).repeat(num_seqs, 1).view(-1)
            ptr_loss = self.ptr_criterion(out_prob, pointer_targets)
            self.total_score += (1 - self.alpha) * ptr_loss / batch_size
            
    def forward(self, all_edus, tree_graph, adj_matrix, roots, rels):
       
        self.total_score = 0
        # Add embedding for root node
        total_loss = self.directed_tree_loss(all_edus, adj_matrix, tree_graph, roots)
        total_loss += self.rels_loss(all_edus, tree_graph, rels)
        return total_loss
    
    def rels_loss(self, all_edus, tree_graph, rels):
        parents_ids = tree_graph.edges()[1]
        children_ids = tree_graph.edges()[0]
        parents = all_edus[:, parents_ids, :]
        children = all_edus[:, children_ids, :]
        output = self.rel_clf(th.cat([parents, children], dim=2)).squeeze(0) # relation predictions
        # print("OUTPUT ", output.shape, output)
        return self.ce_loss(output, rels[rels!=0].to("cuda:0"))
    
    def directed_tree_loss(self, all_edus, adj_matrix, tree_graph, root):

        batch, seq_len, _ = all_edus.shape
        all_edus = all_edus.squeeze(0)
        doc_embed = th.mean(all_edus, dim=0)
        tree_graph.ndata['h'] = all_edus
        tree_graph.ndata['ch_h'] = th.zeros_like(all_edus)
        tree_graph.register_message_func(self.message_func)
        tree_graph.register_reduce_func(lambda x: self.reduce_func(x, doc_embed, batch, seq_len))    
        tree_graph.pull(tree_graph.nodes())
                            
        compat_matrix = self.get_compat_matrix(all_edus)
        root_scores = self.root_clf(all_edus).view(all_edus.shape[0], -1)
        all_edus = all_edus.unsqueeze(0)
        self.total_score += self.logistic_loss(compat_matrix, 
                                               adj_matrix, 
                                               (root_scores, root))
        return self.total_score
        
    def split_node_embed(self, h_cat, doc_lengths):
        # Extract node sequences (without padding on the right)
        sample_node_embeds = []
        for i, doc_len in enumerate(doc_lengths):
            sample_node_embeds.append(h_cat[i, 0:doc_len])
        return sample_node_embeds
    
    def build_target_tensor(self, doc_lengths, seq_len):
        # Batched labels for pointer
        batch_size = len(doc_lengths)
        target_tensor =  th.ones((batch_size, seq_len), device=self.config[DEVICE],dtype=th.long) * -100
        for i, doc_len in enumerate(doc_lengths):
            target_tensor[i, 0:doc_len] = th.arange(doc_len)
        return target_tensor
            
    def get_compat_matrix(self, h_cat):
        # Scores for N nodes -> NxNx2 compatibility matrix
        # entry i(0, N-1), j(0, N-1), k(0,1) means score for 
        # node i being k'th parent (left or right) of node j
        h_cat = h_cat.unsqueeze(0)
        num_nodes = h_cat.shape[1]
        # [1, 2, 3, ...,N] -> [1,1,1...,2,2,2...,N,N,N] (N times)
        src = h_cat.repeat_interleave(num_nodes, 1)
        # [1, 2, 3, ...,N] -> [1, 2, ..., N, 1, 2, ..., N,...] (N times)
        dst = h_cat.repeat(1, num_nodes, 1)
        raw_scores = self.link_predictor(src, dst).view(-1, num_nodes, num_nodes, 2)
        return raw_scores
    
    def bilin_arc_score(self, src, dst):
        # xWy + Wx + Wy + b
        bilin_score = self.bilin_arc(src, dst)
        head_score = self.head_lin_arc(src)
        dep_score = self.dep_lin_arc(dst)
        return bilin_score + head_score + dep_score
        
    def logistic_loss(self, compat_matrix, adj_matrix, root):
        if self.training:
            left_adj_matrix, right_adj_matrix = adj_matrix
            # print("Train ", left_adj_matrix.shape, right_adj_matrix.shape)
        else:
            compat_matrix = compat_matrix
            # print("VAL C ", compat_matrix.shape)
            # adj_matrix = adj_matrix.unsqueeze(0)
            left_adj_matrix, right_adj_matrix = adj_matrix #[:,:,:,0], adj_matrix[:,:,:,1]
            # print("Val ", left_adj_matrix.shape, right_adj_matrix.shape)

        # print("ROOT ", root)
        root_scores, true_root = root    
        # print("TRUE ROOT ", true_root)
        num_nodes = root_scores.shape[1]
        # Convert to double for numerical stability
        root_scores = root_scores.double()
        compat_matrix = compat_matrix.double()
        # Scores for root selection
        index = th.arange(root_scores.shape[0], device="cuda:0")
        # print("SCORES ", root_scores.shape, root_scores, root)
        gold_tree_weight = root_scores[true_root]
        # Scores for left edges
        left_edge_compat = compat_matrix[:, :, :, 0]
        gold_tree_weight += th.sum(left_edge_compat * left_adj_matrix.squeeze(-1), dim=(1,2))
        # Scores for right edges
        right_edge_compat = compat_matrix[:, :, :, 1]
        gold_tree_weight += th.sum(right_edge_compat * right_adj_matrix.squeeze(-1), dim=(1,2))
        # Computing Z
        A = th.exp(compat_matrix)
        root_scores = th.exp(root_scores)
        A = th.sum(A, dim=3)    
        laplacian = th.diag_embed(th.sum(A, dim=1)) - A
        # Replacing top row with root scores (see paper)
        laplacian[:, 0, :] = root_scores.view(1, -1)
        # Negative log likelihood
        logdet = th.logdet(laplacian)
        # Ignore unstable cases (happens for short documents
        #                        late into training)
        mask = (gold_tree_weight <= logdet * 
                (th.isnan(gold_tree_weight) != 1) * 
                (th.isnan(logdet) != 1)).long()
        # loss = log Z - log e^(score of gold tree)
        loss = (logdet - gold_tree_weight) * mask
        loss[th.isnan(loss) == 1] = 0
        return self.alpha * th.sum(loss)

    def decode(self, all_edus, trees_graph, gold_tree):
        
        l_trees_graph, r_trees_graph, roots = trees_graph
        trees_graph = build_trees_graph(l_trees_graph, r_trees_graph)
        left_adj_matrix = l_trees_graph.reverse() \
                            .adjacency_matrix(transpose=True, ctx=th.device(self.config[DEVICE])) \
                            .to_dense().unsqueeze(2) 
        right_adj_matrix = r_trees_graph.reverse() \
                            .adjacency_matrix(transpose=True, ctx=th.device(self.config[DEVICE])) \
                            .to_dense().unsqueeze(2)
        adj_matrix = th.cat([left_adj_matrix,right_adj_matrix], dim=2)
        return self.decode_directed(all_edus, gold_tree, adj_matrix, roots, trees_graph)

    def decode_directed(self, all_edus, gold_tree, gold_adj_matrix, gold_root, trees_graph):
        
        h_cat, doc_embed, num_seqs = self.edu_embed_model([all_edus])
        h_cat = h_cat.squeeze(0)
        num_nodes = int(h_cat.shape[0])        
        # Scores for parenthood and rootness
        compat_matrix_full = self.get_compat_matrix(h_cat.unsqueeze(0)).squeeze()
        root_scores = self.root_clf(h_cat).view(-1)
        # Decode the tree structure
        msp_result, etype, root = self.decode_mst(compat_matrix_full, root_scores)
        # Decode the EDU order from the tree
        dep_tree_root, new_adj_matrix = self.arrange_dep_tree_rootclf(msp_result, etype, int(root))
        uas, las = calc_uas_las(new_adj_matrix, gold_adj_matrix, root, gold_root, num_nodes)
        embeds = (h_cat, doc_embed)
        pred_order = self.node_order(embeds, dep_tree_root)
        pred_order = th.tensor(pred_order, device="cuda:0")      
        # Make 1 to num_nodes instead of 0 to num_nodes - 1
        pred_order += 1
        # Computing validation loss
        log_loss = self.logistic_loss(compat_matrix_full, gold_adj_matrix, (root_scores.unsqueeze(0), gold_root))
        self.total_score = 0
        trees_graph.ndata['h'] = h_cat
        trees_graph.ndata['ch_h'] = th.zeros_like(h_cat)
        trees_graph.register_message_func(self.message_func)
        trees_graph.register_reduce_func(lambda x: self.reduce_func(x, doc_embed, 1, num_nodes))  
        # Loss is accumulated in self.total_score
        trees_graph.pull(trees_graph.nodes())
        del trees_graph.ndata['h']
        del trees_graph.ndata['ch_h']
        return pred_order, self.total_score + log_loss, uas, las

    def decode_mst(self, compat_matrix_full, root_scores):
        
        num_nodes = int(root_scores.shape[0])
        beam_size = min(num_nodes, 5)
        arcs = []
        
        compat_matrix, etype = th.max(compat_matrix_full.squeeze(0), dim=2)
        
        _, indices = th.topk(root_scores, beam_size)
        
        for i in range(0, num_nodes):
            for j in range(0, num_nodes):
                if i != j:
                    arcs.append(Arc(j, float(-compat_matrix[i,j]), i))
                    
        # Find best tree                
        candidate_trees = []            
        tree_scores = []
        for root_idx in range(beam_size):
            root_id = int(indices[root_idx])
            msp_result = min_spanning_arborescence(arcs, root_id)
            score = 0
            for node in msp_result.values():
                score += node.weight
            candidate_trees.append(msp_result)
            tree_scores.append(score)
            
        best_idx = np.argmin(np.array(tree_scores))
        msp_result = candidate_trees[best_idx]
        
        return msp_result, etype, indices[best_idx]
    
    def node_order(self, embed, root):
        acc = []
        for lnode in root.lnodes:
            acc.append(self.node_order(embed, lnode))
            
        if len(root.lnodes) > 1:
            acc = self.direct_ch_order(embed, root.lnodes, acc, root.edu_id)
        elif len(root.lnodes) == 1:
            acc = acc[0]
                
        acc.append(root.edu_id)
            
        right_acc = []
                
        for rnode in root.rnodes:
            right_acc.append(self.node_order(embed, rnode))
            
        if len(root.rnodes) > 1:
            right_acc = self.direct_ch_order(embed, root.rnodes, right_acc, root.edu_id)
        elif len(root.rnodes) == 1:
            right_acc = right_acc[0]
            
        acc.extend(right_acc)
        
        return acc

    def direct_ch_order(self, embed, children, acc, root_id):
        
        h_cat, doc_embed = embed
        h_cat = h_cat.squeeze(0)
        nodes_indices = th.tensor([node.edu_id for node in children], device="cuda:0")
        # print("HCAT ", h_cat.shape)
        # print("DOC EMBED ", doc_embed.shape, doc_embed)
        # print("NODE INDICES ", nodes_indices)
        inputs = h_cat[nodes_indices]
        inputs = inputs.unsqueeze(0)
        pred_order = self.pointer_net.decode(inputs, doc_embed)
        ordered_acc = [] 
        for i in pred_order.squeeze(0):
            ordered_acc.extend(acc[i])
            
        return ordered_acc
            
    def arrange_dep_tree_rootclf(self, result_dict, etype, root):
        nodes = [0] * etype.shape[0]
        num_nodes = len(result_dict) + 1
        new_adj_matrix = th.zeros((num_nodes, num_nodes, 2), 
                                  dtype=th.long, 
                                  device="cuda:0")

        for _, value in result_dict.items():
            new_adj_matrix[value.head, value.tail, etype[value.head, value.tail]] = 1
            if nodes[value.head] == 0:
                nodes[value.head] = DepSpanNode(value.head)
            if nodes[value.tail] == 0:
                nodes[value.tail] = DepSpanNode(value.tail)
            if etype[value.head, value.tail]:
                nodes[value.head].rnodes.append(nodes[value.tail])
            else:
                nodes[value.head].lnodes.append(nodes[value.tail])
        # assert nodes[root] != 0
        # print(root)
        return nodes[root], new_adj_matrix   

def calc_uas_las(new_adj_matrix, gold_adj_matrix,
                root, gold_root, num_nodes):
    flat_new_adj = th.sum(new_adj_matrix, dim=2)
    flat_gold_adj = th.sum(gold_adj_matrix, dim=2)
    uas = 1 - th.sum(flat_new_adj != flat_gold_adj).float() / (num_nodes * 2) - int(root != gold_root) / num_nodes
    las = 1 - th.sum(new_adj_matrix != gold_adj_matrix).float() / (num_nodes * 2) - int(root != gold_root) / num_nodes
    
    return uas, las

def build_trees_graph(l_trees_graph, r_trees_graph):
    trees_graph = dgl.DGLGraph()
    trees_graph.add_nodes(l_trees_graph.nodes().shape[0])
    trees_graph.add_edges(l_trees_graph.edges()[0], l_trees_graph.edges()[1])
    trees_graph.add_edges(r_trees_graph.edges()[0], r_trees_graph.edges()[1])
    return trees_graph
