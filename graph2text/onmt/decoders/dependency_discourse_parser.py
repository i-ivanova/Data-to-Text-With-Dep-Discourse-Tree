import torch as th
import torch.nn as nn
import dgl
import numpy as np

from models.ordered_edu_embedding import OrderedEduEmbedding
from models.modules.mst import *
from dataset.utils.constants import *
from dataset.utils.other import *

from collections import defaultdict
import copy
import gc

class DependencyDiscourseParser(nn.Module):
    
    def __init__(self, config):
        
        super(DependencyDiscourseParser, self).__init__()
        self.device = config[DEVICE]
        self.hid_dim = config[LSTM_HID]
        self.config = config
                
        self.bilin_arc = nn.Bilinear(self.hid_dim, self.hid_dim, 1)
        self.head_lin_arc = nn.Linear(self.hid_dim, 1, bias=False)
        self.dep_lin_arc = nn.Linear(self.hid_dim, 1, bias=False)
                
        self.link_predictor = self.bilin_arc_score        
        
        self.root_clf = nn.Sequential(
                nn.Linear(self.hid_dim, 1)
        )
        self.rel_clf = nn.Sequential(
                nn.Linear(self.hid_dim*2, self.hid_dim),
                nn.GELU(),
                nn.Linear(self.hid_dim, 18)
        )
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.edu_embed_model = OrderedEduEmbedding(config)
        
                    
    def forward(self, tree):
        
        # Add embedding for root node
        self.total_score = 0
        edu_embeds = self.edu_embed_model(tree)
        edu_embeds = edu_embeds.unsqueeze(0)
                    
        compat_matrix = self.get_compat_matrix(edu_embeds)
        # Loss for arcs
        root_scores = self.root_clf(edu_embeds).view(edu_embeds.shape[1])
        self.total_score += self.arc_loss(compat_matrix, root_scores, tree)
        # Loss for relations
        self.total_score += self.rel_loss(edu_embeds, tree)
        return self.total_score
                        
    def get_compat_matrix(self, h_cat):
        # Scores for N nodes -> NxNx2 compatibility matrix
        # entry i(0, N-1), j(0, N-1), k(0,1) means score for 
        # node i being k'th parent (left or right) of node j
        num_nodes = h_cat.shape[1]
        # [1, 2, 3, ...,N] -> [1,1,1...,2,2,2...,N,N,N] (N times)
        src = h_cat.repeat_interleave(num_nodes, 1)
        # [1, 2, 3, ...,N] -> [1, 2, ..., N, 1, 2, ..., N,...] (N times)
        dst = h_cat.repeat(1, num_nodes, 1)
        raw_scores = self.link_predictor(src, dst).view(-1, num_nodes, num_nodes, 1)
        return raw_scores
    
    def bilin_arc_score(self, src, dst):
        # xWy + Wx + Wy + b
        bilin_score = self.bilin_arc(src, dst)
        head_score = self.head_lin_arc(src)
        dep_score = self.dep_lin_arc(dst)
        return bilin_score + head_score + dep_score
        
    def arc_loss(self, compat_matrix, root_scores, tree):
        adj_matrix, true_root = tree.adj_matrix, tree.root_id
        num_nodes = root_scores.shape[0]
        # Convert to double for numerical stability
        root_scores = root_scores.double()
        compat_matrix = compat_matrix.double()
        # Scores for root selection
        gold_tree_weight = root_scores[true_root]
        edge_compat = compat_matrix[0, :, :, 0]
        gold_tree_weight += th.sum(edge_compat * adj_matrix.to(self.config[DEVICE]))
        # Computing Z
        A = th.exp(compat_matrix)
        root_scores = th.exp(root_scores)
        A = th.sum(A, dim=3)    
        laplacian = th.diag_embed(th.sum(A, dim=1)) - A
        # Replacing top row with root scores (see paper)
        laplacian[:, 0, :] = root_scores
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
        return th.sum(loss)
    
    def rel_loss(self, edu_embeds, tree):
        parents_ids = th.tensor(tree.parents)
        relation_ids = th.tensor(tree.relations)
        
        children = edu_embeds[:, parents_ids != -1, :]
        parents = edu_embeds[:, parents_ids[parents_ids != -1], :]
        gold_relations = relation_ids[parents_ids != -1]
        
        rel_embeds = th.cat([parents, children], dim=2)

        rel_scores = self.rel_clf(rel_embeds).squeeze(0)
        return self.ce_loss(rel_scores, gold_relations.to(self.config[DEVICE]))
    
    def decode(self, tree):
        return self.decode_directed(tree)

    def predict(self, tree):
        if len(tree.edus) == 1:
            return [(-1, 'root')]
        mst_result, root, rel_idx = self.decode_directed(tree, compute_loss=False)
        rel_idx = rel_idx.squeeze()
        pred_tree = []
        for i in range(len(mst_result) + 1):
            if i in mst_result:
                pred_tree.append((mst_result[i].head, idx2rel[int(rel_idx[i])]))
            else:
                pred_tree.append((-1, 'root'))

        return pred_tree 
    
    def decode_directed(self, tree, compute_loss=False):
        
        edu_embeds = self.edu_embed_model(tree)
        edu_embeds = edu_embeds.unsqueeze(0)
        # Scores for parenthood and rootness
        compat_matrix = self.get_compat_matrix(edu_embeds).squeeze()
        root_scores = self.root_clf(edu_embeds).view(-1)
        # Decode the tree structure
        mst_result, root = self.decode_mst(compat_matrix, root_scores)
        rel_probs, rel_idx = self.decode_relations(mst_result, edu_embeds, root)
        if compute_loss:
            # Decode the EDU order from the tree
            uas, las = calc_uas_las(mst_result, rel_idx, root, tree)
            # Computing validation loss
            log_loss = self.arc_loss(compat_matrix.unsqueeze(0).unsqueeze(-1), root_scores, tree)
            log_loss += self.rel_loss(edu_embeds, tree)
            return log_loss, uas, las#pred_tree, log_loss, uas, las
        else:
            return mst_result, root, rel_idx
        
    def decode_mst(self, compat_matrix, root_scores):
        
        num_nodes = int(root_scores.shape[0])
        beam_size = min(num_nodes, self.config[DEP_BEAM_SIZE])
        arcs = []        
        _, root_indices = th.topk(root_scores, beam_size)
        
        for i in range(0, num_nodes):
            for j in range(0, num_nodes):
                if i != j:
                    arcs.append(Arc(j, float(-compat_matrix[i,j]), i))
                    
        # Find best tree                
        candidate_trees = []            
        tree_scores = []
        for root_idx in range(beam_size):
            root_id = int(root_indices[root_idx])
            msp_result = min_spanning_arborescence(arcs, root_id)
            score = 0
            for node in msp_result.values():
                score += node.weight
            candidate_trees.append(msp_result)
            tree_scores.append(score)
        
        best_idx = np.argmin(np.array(tree_scores))
        mst_result = candidate_trees[best_idx]
        return mst_result, root_indices[best_idx]
    
    def decode_relations(self, mst, edu_embeds, root):
        pred_heads, pred_tails = [], []
        num_edus = edu_embeds.shape[1]
        for i in range(num_edus):
            if i != root:
                pred_heads.append(mst[i].head)
                pred_tails.append(i)
        rel_feats = th.cat([edu_embeds[:,pred_heads], edu_embeds[:,pred_tails]], dim=2)
        rel_probs = self.rel_clf(rel_feats)
        rel_idx = th.argmax(rel_probs, dim=2)
        return rel_probs, th.cat([rel_idx[:,:root], th.tensor([17]).unsqueeze(0).to(self.config[DEVICE]), \
                                  rel_idx[:,root:]], dim=1)

    def adj_matrix_from_mst(self, mst):
        num_nodes = len(mst) + 1
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        for arc in mst.values():
            adj_matrix[arc.head, arc.tail] = 1
        return adj_matrix
    
def calc_uas_las(mst_result, rel_idx, root, tree):
    uas = 0
    las = 0
    num_nodes = len(tree.parents)
    rel_idx = rel_idx.squeeze()
    
    if root == tree.root_id:
        uas += 1
        las += 1
    
    for pred in mst_result.values():
        if pred.head == tree.parents[pred.tail]:
            uas += 1
            if rel_idx[pred.tail] == tree.relations[pred.tail]:
                las += 1
    uas /= num_nodes
    las /= num_nodes
    return uas, las