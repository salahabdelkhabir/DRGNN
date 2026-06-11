import os
import time
import copy
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from drgnn.utils.data_utils import (
    initialize_node_embedding, evaluate_graph_construct,
    get_all_metrics_fb, get_n_params, get_wandb_log_dict,
    print_dict, disable_all_gradients, evaluate_fb,
    evaluate_graphmask, convert2str, Full_Graph_NegSampler,
    Minibatch_NegSampler, get_device
)


class MovingAverage:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []

    def register(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_value(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class LagrangianOptimization:
    def __init__(self, optimizer, device, batch_size_multiplier=None):
        self.optimizer = optimizer
        self.device = device
        self.batch_size_multiplier = batch_size_multiplier
        self.f_lambda = torch.tensor(0.5, device=device, requires_grad=True)
        self.g_lambda = torch.tensor(0.5, device=device, requires_grad=True)

    def update(self, f, g):
        loss = self.f_lambda * f + self.g_lambda * g
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DRGNN(nn.Module):
    def __init__(self, data, weight_bias_track=False, proj_name='DRGNN', exp_name='DRGNN', device=None):
        super(DRGNN, self).__init__()
        if device is None:
            device = get_device()
        self.device = torch.device(device)
        self.weight_bias_track = weight_bias_track
        self.G = data.G
        self.df = data.df
        self.df_train = data.df_train
        self.df_valid = data.df_valid
        self.df_test = data.df_test
        self.data_folder = data.data_folder
        self.disease_eval_idx = data.disease_eval_idx
        self.split = data.split
        self.no_kg = data.no_kg

        self.disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']

        self.dd_etypes = [
            ('drug', 'contraindication', 'disease'),
            ('drug', 'indication', 'disease'),
            ('drug', 'off-label use', 'disease'),
            ('disease', 'rev_contraindication', 'drug'),
            ('disease', 'rev_indication', 'drug'),
            ('disease', 'rev_off-label use', 'drug')
        ]

        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)
            self.wandb = wandb
        else:
            self.wandb = None
        self.config = None

    def model_initialize(self, n_hid=128, n_inp=128, n_out=128, proto=True,
                         proto_num=5, attention=False, sim_measure='all_nodes_profile',
                         bert_measure='disease_name', agg_measure='rarity',
                         exp_lambda=0.7, num_walks=200, walk_mode='bit', path_length=2):
        if self.no_kg and proto:
            print('Ablation study on No-KG. No proto learning is used...')
            proto = False

        self.G = self.G.to('cpu')
        self.G = initialize_node_embedding(self.G, n_inp)
        self.g_valid_pos, self.g_valid_neg = evaluate_graph_construct(
            self.df_valid, self.G, 'fix_dst', 1, self.device)
        self.g_test_pos, self.g_test_neg = evaluate_graph_construct(
            self.df_test, self.G, 'fix_dst', 1, self.device)

        self.config = {
            'n_hid': n_hid, 'n_inp': n_inp, 'n_out': n_out,
            'proto': proto, 'proto_num': proto_num, 'attention': attention,
            'sim_measure': sim_measure, 'bert_measure': bert_measure,
            'agg_measure': agg_measure, 'num_walks': num_walks,
            'walk_mode': walk_mode, 'path_length': path_length
        }

        self.model = HeteroRGCN(
            self.G, in_size=n_inp, hidden_size=n_hid, out_size=n_out,
            attention=attention, proto=proto, proto_num=proto_num,
            sim_measure=sim_measure, bert_measure=bert_measure,
            agg_measure=agg_measure, num_walks=num_walks,
            walk_mode=walk_mode, path_length=path_length,
            split=self.split, data_folder=self.data_folder,
            exp_lambda=exp_lambda, device=self.device
        ).to(self.device)

        self.best_model = self.model

    def pretrain(self, n_epoch=1, learning_rate=1e-3, batch_size=1024,
                 train_print_per_n=20, sweep_wandb=None):
        if self.no_kg:
            raise ValueError(
                'During No-KG ablation, pretraining is infeasible because '
                'it is the same as finetuning...')

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"Running on: {device_name}")

        self.G = self.G.to('cpu')
        print('Creating minibatch pretraining dataloader...')
        train_eid_dict = {
            etype: self.G.edges(form='eid', etype=etype)
            for etype in self.G.canonical_etypes
        }
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

        dataloader = dgl.dataloading.EdgeDataLoader(
            self.G, train_eid_dict, sampler,
            negative_sampler=Minibatch_NegSampler(self.G, 1, 'fix_dst'),
            batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        print('Start pre-training with #param: %d' % get_n_params(self.model))

        for epoch in range(n_epoch):
            total_steps = len(dataloader)
            epoch_start_time = time.time()

            for step, (nodes, pos_g, neg_g, blocks) in enumerate(dataloader):
                step_start_time = time.time()
                current_step = step + 1
                blocks = [i.to(self.device) for i in blocks]
                pos_g = pos_g.to(self.device)
                neg_g = neg_g.to(self.device)

                pred_score_pos, pred_score_neg, pos_score, neg_score = \
                    self.model.forward_minibatch(
                        pos_g, neg_g, blocks, self.G,
                        mode='train', pretrain_mode=True)

                scores = torch.cat((pos_score, neg_score)).reshape(-1,)
                labels_list = [1] * len(pos_score) + [0] * len(neg_score)
                loss = F.binary_cross_entropy(
                    scores, torch.Tensor(labels_list).float().to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.weight_bias_track:
                    self.wandb.log({"Pretraining Loss": loss})

                if step % train_print_per_n == 0:
                    auroc_rel, auprc_rel, micro_auroc, micro_auprc, \
                        macro_auroc, macro_auprc = get_all_metrics_fb(
                        pred_score_pos, pred_score_neg,
                        scores.reshape(-1,).detach().cpu().numpy(),
                        labels_list, self.G, True)

                    if self.weight_bias_track:
                        temp_d = get_wandb_log_dict(
                            auroc_rel, auprc_rel, micro_auroc, micro_auprc,
                            macro_auroc, macro_auprc, "Pretraining")
                        temp_d.update(
                            {"Pretraining LR": optimizer.param_groups[0]['lr']})
                        self.wandb.log(temp_d)

                    if sweep_wandb is not None:
                        sweep_wandb.log({
                            'pretraining_loss': loss,
                            'pretraining_micro_auroc': micro_auroc,
                            'pretraining_macro_auroc': macro_auroc,
                            'pretraining_micro_auprc': micro_auprc,
                            'pretraining_macro_auprc': macro_auprc
                        })

                    print(
                        'Epoch: %d Step: %d/%d LR: %.5f Loss: %.4f, '
                        'Pretrain Micro AUROC: %.4f Pretrain Micro AUPRC: %.4f '
                        'Pretrain Macro AUROC: %.4f Pretrain Macro AUPRC: %.4f'
                        % (epoch, current_step, total_steps,
                           optimizer.param_groups[0]['lr'], loss.item(),
                           micro_auroc, micro_auprc, macro_auroc, macro_auprc))

                    elapsed_time = time.time() - step_start_time
                    remaining_steps = total_steps - current_step
                    estimated_time_remaining = elapsed_time * remaining_steps
                    print(
                        f"Epoch: {epoch} Step: {current_step}/{total_steps} "
                        f"Estimated Time Remaining: "
                        f"{estimated_time_remaining / 60:.2f} minutes")

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time / 60:.2f} minutes.")

        self.best_model = copy.deepcopy(self.model)

    def finetune(self, n_epoch=500, learning_rate=1e-3, train_print_per_n=5,
                 valid_per_n=25, sweep_wandb=None, save_name=None):
        best_val_acc = 0

        epochs = []
        train_losses, valid_losses = [], []
        train_micro_aurocs, train_macro_aurocs = [], []
        valid_micro_aurocs, valid_macro_aurocs = [], []
        learning_rates = []
        train_rel_aurocs = defaultdict(list)
        train_rel_auprcs = defaultdict(list)
        valid_rel_aurocs = defaultdict(list)
        valid_rel_auprcs = defaultdict(list)

        self.G = self.G.to(self.device)
        neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
        torch.nn.init.xavier_uniform_(self.model.w_rels)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', 0.8)

        for epoch in range(n_epoch):
            negative_graph = neg_sampler(self.G)
            pred_score_pos, pred_score_neg, pos_score, neg_score = \
                self.model(self.G, negative_graph,
                           pretrain_mode=False, mode='train')

            pos_score = torch.cat([pred_score_pos[i] for i in self.dd_etypes])
            neg_score = torch.cat([pred_score_neg[i] for i in self.dd_etypes])

            scores = torch.sigmoid(
                torch.cat((pos_score, neg_score)).reshape(-1,))
            labels_list = [1] * len(pos_score) + [0] * len(neg_score)
            loss = F.binary_cross_entropy(
                scores, torch.Tensor(labels_list).float().to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if self.weight_bias_track:
                self.wandb.log({"Training Loss": loss})

            epochs.append(epoch)
            train_losses.append(loss.item())
            learning_rates.append(optimizer.param_groups[0]['lr'])

            if epoch % train_print_per_n == 0:
                auroc_rel, auprc_rel, micro_auroc, micro_auprc, \
                    macro_auroc, macro_auprc = get_all_metrics_fb(
                    pred_score_pos, pred_score_neg,
                    scores.reshape(-1,).detach().cpu().numpy(),
                    labels_list, self.G, True)

                for rel in auroc_rel:
                    train_rel_aurocs[rel].append(auroc_rel[rel])
                    train_rel_auprcs[rel].append(auprc_rel[rel])

                train_micro_aurocs.append(micro_auroc)
                train_macro_aurocs.append(macro_auroc)

                if self.weight_bias_track:
                    temp_d = get_wandb_log_dict(
                        auroc_rel, auprc_rel, micro_auroc, micro_auprc,
                        macro_auroc, macro_auprc, "Training")
                    temp_d.update({"LR": optimizer.param_groups[0]['lr']})
                    self.wandb.log(temp_d)

                print(
                    'Epoch: %d LR: %.5f Loss %.4f, Train Micro AUROC %.4f '
                    'Train Micro AUPRC %.4f Train Macro AUROC %.4f '
                    'Train Macro AUPRC %.4f' % (
                        epoch, optimizer.param_groups[0]['lr'], loss.item(),
                        micro_auroc, micro_auprc, macro_auroc, macro_auprc))
                print('----- AUROC Performance in Each Relation -----')
                print_dict(auroc_rel)
                print('----- AUPRC Performance in Each Relation -----')
                print_dict(auprc_rel)
                print('----------------------------------------------')

            del pred_score_pos, pred_score_neg, scores, labels_list

            if epoch % valid_per_n == 0:
                print('Validation.....')
                (auroc_rel, auprc_rel, micro_auroc, micro_auprc,
                 macro_auroc, macro_auprc), loss = evaluate_fb(
                    self.model, self.g_valid_pos, self.g_valid_neg,
                    self.G, self.dd_etypes, self.device, mode='valid')

                valid_losses.append(loss)
                valid_micro_aurocs.append(micro_auroc)
                valid_macro_aurocs.append(macro_auroc)

                for rel in auroc_rel:
                    valid_rel_aurocs[rel].append(auroc_rel[rel])
                    valid_rel_auprcs[rel].append(auprc_rel[rel])

                if best_val_acc < macro_auroc:
                    best_val_acc = macro_auroc
                    self.best_model = copy.deepcopy(self.model)

        plt.figure(figsize=(20, 15))
        plt.subplot(3, 2, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(range(0, n_epoch, valid_per_n), valid_losses,
                 label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(range(0, n_epoch, train_print_per_n), train_micro_aurocs,
                 label='Training Micro AUROC')
        plt.plot(range(0, n_epoch, valid_per_n), valid_micro_aurocs,
                 label='Validation Micro AUROC')
        plt.xlabel('Epoch')
        plt.ylabel('Micro AUROC')
        plt.title('Training and Validation Micro AUROC')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(range(0, n_epoch, train_print_per_n), train_macro_aurocs,
                 label='Training Macro AUROC')
        plt.plot(range(0, n_epoch, valid_per_n), valid_macro_aurocs,
                 label='Validation Macro AUROC')
        plt.xlabel('Epoch')
        plt.ylabel('Macro AUROC')
        plt.title('Training and Validation Macro AUROC')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(epochs, learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

        plt.subplot(3, 2, 5)
        for rel in train_rel_aurocs:
            plt.plot(range(0, n_epoch, train_print_per_n),
                     train_rel_aurocs[rel],
                     label=f'Rel {rel} (Train)', linestyle='-', alpha=0.7)
            plt.plot(range(0, n_epoch, valid_per_n),
                     valid_rel_aurocs[rel],
                     label=f'Rel {rel} (Valid)', linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('AUROC')
        plt.title('Relation-specific AUROC')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.subplot(3, 2, 6)
        for rel in train_rel_auprcs:
            plt.plot(range(0, n_epoch, train_print_per_n),
                     train_rel_auprcs[rel],
                     label=f'Rel {rel} (Train)', linestyle='-', alpha=0.7)
            plt.plot(range(0, n_epoch, valid_per_n),
                     valid_rel_auprcs[rel],
                     label=f'Rel {rel} (Valid)', linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('AUPRC')
        plt.title('Relation-specific AUPRC')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        if save_name is not None:
            plt.savefig(f"{save_name}_training_curves.png",
                        bbox_inches='tight', dpi=300)
        plt.show()

    def save_model(self, path: str):
        if not os.path.exists(path):
            os.mkdir(path)
        if self.config is None:
            raise ValueError('No model is initialized...')
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        torch.save(self.best_model.state_dict(),
                   os.path.join(path, 'model.pt'))

    def predict(self, df):
        out = {}
        g = self.G
        df_in = df[['x_idx', 'relation', 'y_idx']]
        for etype in g.canonical_etypes:
            try:
                df_temp = df_in[df_in.relation == etype[1]]
            except KeyError:
                print(etype[1])
            src = (torch.Tensor(df_temp.x_idx.values)
                   .to(self.device).to(dtype=torch.int64))
            dst = (torch.Tensor(df_temp.y_idx.values)
                   .to(self.device).to(dtype=torch.int64))
            out.update({etype: (src, dst)})
        g_eval = dgl.heterograph(
            out, num_nodes_dict={
                ntype: g.number_of_nodes(ntype) for ntype in g.ntypes})
        g_eval = g_eval.to(self.device)
        g = g.to(self.device)
        self.model.eval()
        pred_score_pos, pred_score_neg, pos_score, neg_score = self.model(
            g, g_eval, g_eval, pretrain_mode=False, mode='test')
        return pred_score_pos

    def retrieve_embedding(self, path=None):
        self.G = self.G.to(self.device)
        h = self.model(self.G, self.G, return_h=True)
        for i, j in h.items():
            h[i] = j.detach().cpu()
        if path is not None:
            with open(os.path.join(path, 'node_emb.pkl'), 'wb') as f:
                pickle.dump(h, f)
        return h

    def retrieve_sim_diseases(self, relation: str, k: int = 5, path=None):
        if relation not in ['indication', 'contraindication', 'off-label']:
            raise ValueError(
                "Please select: 'indication', 'contraindication', 'off-label'")
        etypes = self.dd_etypes
        out_degrees = {}
        in_degrees = {}
        for etype in etypes:
            out_degrees[etype] = torch.where(
                self.G.out_degrees(etype=etype) != 0)
            in_degrees[etype] = torch.where(
                self.G.in_degrees(etype=etype) != 0)
        sim_all_etypes = self.model.pred.sim_all_etypes
        diseaseid2id_etypes = self.model.pred.diseaseid2id_etypes
        id2diseaseid_etypes = {}
        for etype_val, diseaseid2id in diseaseid2id_etypes.items():
            id2diseaseid_etypes[etype_val] = {
                j: i for i, j in diseaseid2id.items()}
        h = self.retrieve_embedding()
        etype_map = {
            'indication': ('disease', 'rev_indication', 'drug'),
            'contraindication': ('disease', 'rev_contraindication', 'drug'),
            'off-label': ('disease', 'rev_off-label use', 'drug')
        }
        etype = etype_map[relation]
        src_type = etype[0]
        src_rel_idx = out_degrees[etype]
        src_h = h[src_type][src_rel_idx]
        h_disease = {
            'disease_query': src_h,
            'disease_query_id': src_rel_idx
        }
        sim = sim_all_etypes[etype][
            np.array([diseaseid2id_etypes[etype][i.item()]
                      for i in h_disease['disease_query_id'][0]])]
        similar_diseases = torch.topk(sim, k + 1).indices[:, 1:]
        similar_diseases = similar_diseases.apply_(
            lambda x: id2diseaseid_etypes[etype][x])
        if path is not None:
            with open(os.path.join(path, 'sim_diseases.pkl'), 'wb') as f:
                pickle.dump(similar_diseases, f)
        return similar_diseases

    def load_pretrained(self, path: str):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        self.model_initialize(**config)
        self.config = config
        state_dict = torch.load(
            os.path.join(path, 'model.pt'), map_location=torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model

    def train_graphmask(self, relation='indication', learning_rate=3e-4,
                        allowance=0.005, epochs_per_layer=1000,
                        penalty_scaling=1, moving_average_window_size=100,
                        valid_per_n=5, no_base=False, gate_hidden_size=32):
        self.relation = relation
        if relation not in ['indication', 'contraindication', 'off-label']:
            raise ValueError(
                "Please select: 'indication', 'contraindication', 'off-label'")

        etype_map = {
            'indication': [('drug', 'indication', 'disease'),
                           ('disease', 'rev_indication', 'drug')],
            'contraindication': [('drug', 'contraindication', 'disease'),
                                 ('disease', 'rev_contraindication', 'drug')],
            'off-label': [('drug', 'off-label use', 'disease'),
                          ('disease', 'rev_off-label use', 'drug')]
        }
        etypes_train = etype_map[relation]
        best_loss_sum = 100

        if "graphmask_model" not in self.__dict__:
            self.graphmask_model = copy.deepcopy(self.best_model)
            self.best_graphmask_model = copy.deepcopy(self.graphmask_model)
            self.graphmask_model.add_graphmask_parameters(
                self.G, gate_hidden_size=gate_hidden_size)
        else:
            print("Training from checkpoint/pretrained model...")

        self.graphmask_model.eval()
        disable_all_gradients(self.graphmask_model)
        optimizer = torch.optim.Adam(
            self.graphmask_model.parameters(), lr=learning_rate)
        self.graphmask_model.to(self.device)
        lagrangian_optimization = LagrangianOptimization(
            optimizer, self.device, batch_size_multiplier=None)
        f_moving_average = MovingAverage(
            window_size=moving_average_window_size)
        g_moving_average = MovingAverage(
            window_size=moving_average_window_size)
        neg_sampler = Full_Graph_NegSampler(
            self.G, 1, 'fix_dst', self.device)
        loss_fct = nn.MSELoss()
        self.G = self.G.to(self.device)

        for layer in reversed(
                list(range(self.graphmask_model.count_layers()))):
            self.graphmask_model.enable_layer(layer)
            for epoch in range(epochs_per_layer):
                self.graphmask_model.train()
                neg_graph = neg_sampler(self.G)
                original_predictions_pos, original_predictions_neg, _, _ = \
                    self.graphmask_model.graphmask_forward(
                        self.G, self.G, neg_graph, graphmask_mode=False,
                        only_relation=relation, no_base=no_base)
                pos_score = torch.cat(
                    [original_predictions_pos[i] for i in etypes_train])
                neg_score = torch.cat(
                    [original_predictions_neg[i] for i in etypes_train])
                original_predictions = torch.sigmoid(
                    torch.cat((pos_score, neg_score))).to('cpu')

                (updated_predictions_pos, updated_predictions_neg,
                 penalty, num_masked) = \
                    self.graphmask_model.graphmask_forward(
                        self.G, self.G, neg_graph, graphmask_mode=True,
                        only_relation=relation, no_base=no_base)
                pos_score = torch.cat(
                    [updated_predictions_pos[i] for i in etypes_train])
                neg_score = torch.cat(
                    [updated_predictions_neg[i] for i in etypes_train])
                updated_predictions = torch.sigmoid(
                    torch.cat((pos_score, neg_score)))
                labels_list = [1] * len(pos_score) + [0] * len(neg_score)
                loss_pred = F.binary_cross_entropy(
                    updated_predictions,
                    torch.Tensor(labels_list).float().to(self.device)).item()
                original_predictions = original_predictions.to(self.device)
                loss_pred_ori = F.binary_cross_entropy(
                    original_predictions,
                    torch.Tensor(labels_list).float().to(self.device)).item()
                loss = loss_fct(original_predictions, updated_predictions)
                g_val = torch.relu(loss - allowance).mean()
                f_val = penalty * penalty_scaling
                lagrangian_optimization.update(f_val, g_val)
                f_moving_average.register(float(f_val.item()))
                g_moving_average.register(float(loss.mean().item()))
                print(
                    "Running epoch {0:n} of GraphMask training. "
                    "Mean divergence={1:.4f}, mean penalty={2:.4f}, "
                    "bce_update={3:.4f}, bce_original={4:.4f}, "
                    "num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
                        epoch, g_moving_average.get_value(),
                        f_moving_average.get_value(), loss_pred,
                        loss_pred_ori,
                        num_masked[0] / self.G.number_of_edges(),
                        num_masked[1] / self.G.number_of_edges()))
                del (original_predictions, updated_predictions,
                     f_val, g_val, loss, pos_score, neg_score,
                     loss_pred_ori, loss_pred, neg_graph)

                if epoch % valid_per_n == 0:
                    loss_sum = evaluate_graphmask(
                        self.graphmask_model, self.G, self.g_valid_pos,
                        self.g_valid_neg, relation, epoch,
                        mode='validation', allowance=allowance,
                        penalty_scaling=penalty_scaling,
                        etypes_train=etypes_train, device=self.device,
                        weight_bias_track=self.weight_bias_track,
                        wandb=self.wandb, no_base=no_base)
                    if loss_sum < best_loss_sum:
                        best_loss_sum = loss_sum
                        self.best_graphmask_model = copy.deepcopy(
                            self.graphmask_model)

        loss_sum, metrics = evaluate_graphmask(
            self.best_graphmask_model, self.G, self.g_test_pos,
            self.g_test_neg, relation, epoch, mode='testing',
            allowance=allowance, penalty_scaling=penalty_scaling,
            etypes_train=etypes_train, device=self.device,
            weight_bias_track=self.weight_bias_track, wandb=self.wandb,
            no_base=no_base)
        if self.weight_bias_track:
            self.wandb.log(metrics)
        return metrics

    def save_graphmask_model(self, path: str):
        if not os.path.exists(path):
            os.mkdir(path)
        if self.config is None:
            raise ValueError('No model is initialized...')
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        torch.save(self.best_graphmask_model.state_dict(),
                   os.path.join(path, 'graphmask_model.pt'))

    def load_pretrained_graphmask(self, path: str, threshold=0.5,
                                   remove_key_parts=False, use_top_k=False,
                                   k=0.05, gate_hidden_size=32):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        self.model_initialize(**config)
        self.config = config
        if "graphmask_model" not in self.__dict__:
            self.graphmask_model = copy.deepcopy(self.best_model)
            self.best_graphmask_model = copy.deepcopy(self.graphmask_model)
            self.graphmask_model.add_graphmask_parameters(
                self.G, threshold, remove_key_parts, use_top_k, k,
                gate_hidden_size)
        state_dict = torch.load(
            os.path.join(path, 'graphmask_model.pt'),
            map_location=torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.graphmask_model.load_state_dict(state_dict)
        self.graphmask_model = self.graphmask_model.to(self.device)
        self.best_graphmask_model = self.graphmask_model

    def retrieve_gates_scores_penalties(self, relation, no_base=False):
        self.g_test_pos, self.g_test_neg = evaluate_graph_construct(
            self.df_test, self.G, 'fix_dst', 1, self.device)
        self.G = self.G.to(self.device)
        self.graphmask_model = self.graphmask_model.eval()
        neg_sampler = Full_Graph_NegSampler(
            self.G, 1, 'fix_dst', self.device)
        neg_graph = neg_sampler(self.G)
        (original_predictions_pos, original_predictions_neg,
         _, _) = self.graphmask_model.graphmask_forward(
            self.G, self.G, neg_graph, graphmask_mode=False,
            only_relation=relation, no_base=no_base)
        (updated_predictions_pos, updated_predictions_neg,
         penalty, num_masked) = self.graphmask_model.graphmask_forward(
            self.G, self.G, neg_graph, graphmask_mode=True,
            only_relation=relation, return_gates=True, no_base=no_base)
        gates = self.graphmask_model.get_gates()
        scores = self.graphmask_model.get_gates_scores()
        penalties = self.graphmask_model.get_gates_penalties()
        whole_graph = [original_predictions_pos, original_predictions_neg,
                       updated_predictions_pos, updated_predictions_neg,
                       num_masked, gates, scores, penalties]
        (original_predictions_pos, original_predictions_neg,
         _, _) = self.graphmask_model.graphmask_forward(
            self.G, self.g_test_pos, self.g_test_neg,
            graphmask_mode=False, only_relation=relation, no_base=no_base)
        (updated_predictions_pos, updated_predictions_neg,
         penalty, num_masked) = self.graphmask_model.graphmask_forward(
            self.G, self.g_test_pos, self.g_test_neg,
            graphmask_mode=True, only_relation=relation,
            return_gates=True, no_base=no_base)
        gates = self.graphmask_model.get_gates()
        scores = self.graphmask_model.get_gates_scores()
        penalties = self.graphmask_model.get_gates_penalties()
        test_graph = [original_predictions_pos, original_predictions_neg,
                      updated_predictions_pos, updated_predictions_neg,
                      num_masked, gates, scores, penalties]
        return whole_graph, test_graph

    def retrieve_save_gates(self, path: str):
        _, scores, _ = self.retrieve_gates_scores_penalties()
        df_raw = pd.read_csv(os.path.join(self.data_folder, 'kg.csv'))
        df = self.df
        df_raw['x_id'] = df_raw.x_id.apply(lambda x: convert2str(x))
        df_raw['y_id'] = df_raw.y_id.apply(lambda x: convert2str(x))
        df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
        df['y_id'] = df.y_id.apply(lambda x: convert2str(x))
        idx2id_all = {}
        id2name_all = {}
        for node_type in self.G.ntypes:
            idx2id = dict(
                df[df.x_type == node_type][['x_idx', 'x_id']].values)
            idx2id.update(
                dict(df[df.y_type == node_type][['y_idx', 'y_id']].values))
            id2name = dict(
                df_raw[df_raw.x_type == node_type][['x_id', 'x_name']].values)
            id2name.update(
                dict(df_raw[df_raw.y_type == node_type][['y_id', 'y_name']].values))
            idx2id_all[node_type] = idx2id
            id2name_all[node_type] = id2name
        all_att_df = pd.DataFrame()
        G = self.G.to('cpu')
        for etypes in G.canonical_etypes:
            etype = etypes[1]
            src_type, dst_type = etypes[0], etypes[2]
            df_temp = pd.DataFrame()
            df_temp['x_idx'] = G.edges(etype=etype)[0].numpy()
            df_temp['y_idx'] = G.edges(etype=etype)[1].numpy()
            df_temp['x_id'] = df_temp['x_idx'].apply(
                lambda x: idx2id_all[src_type][x])
            df_temp['y_id'] = df_temp['y_idx'].apply(
                lambda x: idx2id_all[dst_type][x])
            df_temp['x_name'] = df_temp['x_id'].apply(
                lambda x: id2name_all[src_type][x])
            df_temp['y_name'] = df_temp['y_id'].apply(
                lambda x: id2name_all[dst_type][x])
            df_temp['x_type'] = src_type
            df_temp['y_type'] = dst_type
            df_temp['relation'] = etype
            df_temp[self.relation + '_layer1_att'] = scores[0][etype].reshape(-1,)
            df_temp[self.relation + '_layer2_att'] = scores[1][etype].reshape(-1,)
            all_att_df = all_att_df._append(df_temp)
        all_att_df.to_pickle(
            os.path.join(path, 'graphmask_output_' + self.relation + '.pkl'))
        return all_att_df


class HeteroRGCN(nn.Module):
    def __init__(self, g, in_size, hidden_size, out_size, attention=False,
                 proto=True, proto_num=5, sim_measure='all_nodes_profile',
                 bert_measure='disease_name', agg_measure='rarity',
                 num_walks=200, walk_mode='bit', path_length=2,
                 split=None, data_folder=None, exp_lambda=0.7, device='cpu'):
        super(HeteroRGCN, self).__init__()
        self.device = device
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.w_rels = nn.Parameter(
            torch.empty(len(g.etypes), in_size, out_size))
        nn.init.xavier_uniform_(self.w_rels)

        self.layer_norm = nn.LayerNorm(out_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, neg_g, pretrain_mode=False, mode='train', return_h=False):
        if return_h:
            return {ntype: torch.randn(g.number_of_nodes(ntype), self.out_size)
                    for ntype in g.ntypes}
        pred_score_pos = {}
        pred_score_neg = {}
        for etype in g.canonical_etypes:
            pred_score_pos[etype] = torch.randn(g.number_of_edges(etype), 1)
            pred_score_neg[etype] = torch.randn(g.number_of_edges(etype), 1)
        pos_score = torch.cat(
            [pred_score_pos[i] for i in g.canonical_etypes])
        neg_score = torch.cat(
            [pred_score_neg[i] for i in g.canonical_etypes])
        return pred_score_pos, pred_score_neg, pos_score, neg_score

    def forward_minibatch(self, pos_g, neg_g, blocks, g,
                          mode='train', pretrain_mode=True):
        pred_score_pos = {}
        pred_score_neg = {}
        for etype in pos_g.canonical_etypes:
            pred_score_pos[etype] = torch.randn(
                pos_g.number_of_edges(etype), 1).to(self.device)
            pred_score_neg[etype] = torch.randn(
                neg_g.number_of_edges(etype), 1).to(self.device)
        pos_score = torch.cat(
            [pred_score_pos[i] for i in pos_g.canonical_etypes])
        neg_score = torch.cat(
            [pred_score_neg[i] for i in neg_g.canonical_etypes])
        return pred_score_pos, pred_score_neg, pos_score, neg_score

    def graphmask_forward(self, g, pos_g, neg_g, graphmask_mode=False,
                          only_relation=None, return_gates=False, no_base=False):
        pred_score_pos = {}
        pred_score_neg = {}
        for etype in pos_g.canonical_etypes:
            pred_score_pos[etype] = torch.randn(
                pos_g.number_of_edges(etype), 1).to(self.device)
            pred_score_neg[etype] = torch.randn(
                neg_g.number_of_edges(etype), 1).to(self.device)
        penalty = torch.tensor(0.0)
        num_masked = [0, 0]
        if return_gates:
            return (pred_score_pos, pred_score_neg, penalty, num_masked,
                    {}, {}, {})
        return pred_score_pos, pred_score_neg, penalty, num_masked

    def add_graphmask_parameters(self, g, threshold=0.5,
                                  remove_key_parts=False, use_top_k=False,
                                  k=0.05, gate_hidden_size=32):
        pass

    def count_layers(self):
        return 2

    def enable_layer(self, layer):
        pass

    def get_gates(self):
        return {}

    def get_gates_scores(self):
        return [{}, {}]

    def get_gates_penalties(self):
        return {}
