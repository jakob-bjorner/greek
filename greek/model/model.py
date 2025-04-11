#%%
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from greek.dataset.dataset import AwesomeAlignDatasetReturn
from itertools import accumulate
# collate function is deeply related to how the train function works, so I will define them in the same place.
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers import PreTrainedTokenizerBase
from transformers import AutoConfig
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

@dataclass
class CollateFnReturn:
    srcs: List[str]
    tgts: List[str]
    examples_src: torch.Tensor
    examples_tgt: torch.Tensor
    alignment_construction_params: Dict[str, Any]
    bpe2word_map_src: List[List[int]]
    bpe2word_map_tgt: List[List[int]]
    examples_srctgt: torch.Tensor
    bpe2word_map_srctgt: List[List[int]]
    langid_srctgt: torch.Tensor
    input_examples_src_masked: torch.Tensor
    label_examples_src_masked: torch.Tensor
    input_examples_tgt_masked: torch.Tensor
    label_examples_tgt_masked: torch.Tensor
    input_examples_srctgt_mask_tgt: torch.Tensor 
    label_examples_srctgt_mask_tgt: torch.Tensor
    input_examples_srctgt_mask_src: torch.Tensor 
    label_examples_srctgt_mask_src: torch.Tensor
    input_examples_tgtsrc_mask_src: torch.Tensor 
    label_examples_tgtsrc_mask_src: torch.Tensor
    input_examples_tgtsrc_mask_tgt: torch.Tensor 
    label_examples_tgtsrc_mask_tgt: torch.Tensor
    constituent_info_from_srctgt_mask_tgt: Dict
    constituent_info_from_srctgt_mask_src: Dict
    constituent_info_from_tgtsrc_mask_src: Dict
    constituent_info_from_tgtsrc_mask_tgt: Dict
    constituent_info_from_src_masked: Dict
    constituent_info_from_tgt_masked: Dict
    examples_tgtsrc: torch.Tensor
    bpe2word_map_tgtsrc: List[List[int]]
    langid_tgtsrc: torch.Tensor
    psi_examples_srctgt: torch.Tensor
    psi_labels: torch.Tensor
    
    step: int
    total_steps: int
    
    def to(self, device):
        def tree_map(dict_obj):
            for key, val in dict_obj.items():
                if isinstance(val, torch.Tensor):
                    dict_obj[key] = val.to(device)
                elif isinstance(val, Dict):
                    tree_map(val)
            
        tree_map(self.__dict__)
        return self
                
class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean()

class BaseTextAligner(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()
    @abstractmethod
    def get_aligned_word(self, inputs_src, inputs_tgt):
        raise NotImplementedError()


class AwesomeAlignerValMetrics:
    def __init__(self):
        self.total_num_elements = 0
        self.data = dict()
        self.dataset_num_possible_deleted = 0
        self.dataset_num_sure_deleted = 0
    def update(self, other: Dict):
        if len(self.data) != 0:
            assert other.keys() == self.data.keys(), f"must have the same keys to be able to update the values in the metric item, but got {list(other.keys())=} instead of {list(self.data.keys())}"
        # average by default, and then for specific keys, do specific things.

        num_elements = 0
        
        for key, value in other.items():
            if key in ["word_alignments", "impacts", "srcs", "tgts", "gold_possible_word_alignments", "gold_sure_word_alignments",
                       "token_alignments", "token_impacts", "token_srcs", "token_tgts", "gold_possible_token_alignments", "gold_sure_token_alignments", 
                       "classification_loss_per_pair"]:
                self.data[key] = self.data.get(key, []) + value
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu() # .mean().item()
                if num_elements == 0:
                    num_elements = value.size(0)
                else:
                    assert num_elements == value.size(0), "we only allow tensors of batch size for ease of determining the number of elements"
                value = value.sum().item()

            self.data[key] = self.data.get(key, 0) + value # the default is in the case that this is the first update call.
        self.total_num_elements += num_elements
    def add_preprocessing_stats(self, num_possible_deleted, num_sure_deleted):
        '''add the AER relevant metrics which are preprocessing specific. 
        This function was made for accounting for the words which do not exist to align to when a src doens't correspond to any tgt due to swapping or deletion'''
        self.dataset_num_possible_deleted = num_possible_deleted
        self.dataset_num_sure_deleted = num_sure_deleted
    def compute(self, dataset_name: str, logger, should_plot, current_global_step):
        """computes the aggigate metrics for this class to be returned for logging/printing."""
        aggregate_metrics = dict()

        for key in self.data.keys():
            if "classification_loss_per_pair" == key:
                aggregate_metrics["avg_cls_loss"] = sum(self.data[key]) / len(self.data[key])
            elif "loss_per_batch" in key:
                aggregate_metrics["avg_" + key.removesuffix("_per_batch")] = self.data[key] / self.total_num_elements
            elif key not in ["guesses_made_in_possible", "guesses_made", "num_sure", "guesses_made_in_sure", "word_alignments", "impacts", "srcs", "tgts", "gold_possible_word_alignments", "gold_sure_word_alignments",
                             "token_guesses_made_in_possible", "token_guesses_made", "token_num_sure", "token_guesses_made_in_sure", "token_alignments", "token_impacts", "token_srcs", "token_tgts", "gold_possible_token_alignments", "gold_sure_token_alignments"]:
                aggregate_metrics["avg_" + key] = self.data[key] / self.total_num_elements
        # aggregate_metrics["total_num_elements"] = self.total_num_elements
        aggregate_metrics["Precision"] = 0 if self.data["guesses_made"] == 0 else self.data["guesses_made_in_possible"] / self.data["guesses_made"]
        aggregate_metrics["Recall"] = 0 if self.data["num_sure"] == 0 else self.data["guesses_made_in_sure"] / self.data["num_sure"]
        aggregate_metrics["AER"] = 0 if (self.data["guesses_made"] + self.data["num_sure"]) == 0 else 1 - ((self.data["guesses_made_in_possible"] + self.data["guesses_made_in_sure"]) / (self.data["guesses_made"] + self.data["num_sure"]))
        aggregate_metrics["AER_pp_adj"] = 0 if (self.data["guesses_made"] + self.data["num_sure"] + self.dataset_num_sure_deleted) == 0 else 1 - ((self.data["guesses_made_in_possible"] + self.data["guesses_made_in_sure"]) / (self.data["guesses_made"] + self.data["num_sure"] + self.dataset_num_sure_deleted))
        # now on token level:
        aggregate_metrics["token_Precision"] = 0 if self.data["token_guesses_made"] == 0 else self.data["token_guesses_made_in_possible"] / self.data["token_guesses_made"]
        aggregate_metrics["token_Recall"] = 0 if self.data["token_num_sure"] == 0 else self.data["token_guesses_made_in_sure"] / self.data["token_num_sure"]
        aggregate_metrics["token_AER"] = 0 if (self.data["token_guesses_made"] + self.data["token_num_sure"]) == 0 else 1 - ((self.data["token_guesses_made_in_possible"] + self.data["token_guesses_made_in_sure"]) / (self.data["token_guesses_made"] + self.data["token_num_sure"]))
        # TODO: add dataset_num_sure_deleted tokens calculation and constant.
        # aggregate_metrics["token_AER_pp_adj"] = 0 if (self.data["token_guesses_made"] + self.data["token_num_sure"] + self.dataset_num_sure_deleted) == 0 else 1 - ((self.data["token_guesses_made_in_possible"] + self.data["token_guesses_made_in_sure"]) / (self.data["token_guesses_made"] + self.data["token_num_sure"] + self.dataset_num_sure_deleted))
        
        # create the images for uploading
        # need src, tgt, sure_gold, possible_gold, hypothesis, 
        if should_plot: # test plotting with: python run.py +run_modifier=SupervisedRunConfig run_type=supervised_show_all_plots datasetmap@trainer.datasetloaders.val_datasets=nozhSupervisedAwesomeAlignDatasetsMapEval
            def display_alignment(index, sure, possible, word_alignment, source, target, language, p_r_aer=None, bigger_boxes=False, matrix_to_compute_p_r_aer=None):
                m = len(source)
                l = len(target)
                hypothesis_mat = np.zeros((m, l))
                hypothesis_mat[tuple(zip(*word_alignment))] = 1
                # plt.style.use('_mpl-gallery-nogrid')
                multiplier = 1 if bigger_boxes else 0.5 
                multiplier = 0.2 # smaller boxes for wandb
                fig, ax = plt.subplots(figsize = (1 + l * multiplier, 1 + m * multiplier)) # w x h for some reason??
                # precision recall for this image:
                
                num_sure = len(sure)
                guesses_made = len(word_alignment)
                guesses_made_in_sure = len(set(word_alignment).intersection(sure))
                guesses_made_in_possible = len(set(word_alignment).intersection(possible))
                precision = 0 if guesses_made == 0 else guesses_made_in_possible / guesses_made
                recall = 0 if num_sure == 0 else guesses_made_in_sure / num_sure
                aer = 0 if (guesses_made + num_sure) == 0 else 1 - ((guesses_made_in_possible + guesses_made_in_sure) / (guesses_made + num_sure))
                
                num_words_total = m + l
                num_words_covered = len(set(s for s,t in word_alignment)) + len(set(t for s,t in word_alignment))
                coverage = num_words_covered / num_words_total

                ax.set_title(f"{index:3} p_r_aer_cvg=[{precision:.5f}, {recall:.5f}, {aer:.5f}, {coverage:.5f}]")
                im = ax.imshow(hypothesis_mat, cmap='Blues', vmin=0, vmax=1)
                ax.set_xticks(np.arange(len(target)), labels=target)
                
                if 'jaen' not in (language or " "):
                    ax.set_yticks(np.arange(len(source)), labels=source)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                        rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                # for i in range(m): # this will be useful when I want to plot percentages maybe when doing something more than word alignment.
                #     for j in range(l):
                #         text = ax.text(j, i, hypothesis_mat[i, j],
                #                     ha="center", va="center", color="orange")
                patches_sure = []
                patches_possible = []
                for box in sure:
                    patches_sure.append(Rectangle((box[1]-0.3,box[0]-0.3), 0.6, 0.6, angle=45, rotation_point="center"))
                for box in set(possible).difference(set(sure)):
                    patches_possible.append(Rectangle((box[1]-0.4,box[0]-0.4), 0.8, 0.8))
                ax.add_collection(PatchCollection(patches_sure, edgecolor="LightGreen", facecolor="none",lw=2))
                ax.add_collection(PatchCollection(patches_possible, edgecolor="Orange", facecolor="none", lw=2))
                # ax.legend(["Orange is possible", "Green is sure"]) # this doesn't seem to work, but only necessary for paper, and can be added in post.
                fig.tight_layout()
                # print(df.sort_values(out_format + 'impact_on_aer').index.tolist()[::-1])
                return fig 
            
            consistent_index = 0 if len(self.data["srcs"]) < 30 else 29 # could be specified in dataset's config, and retrieved from the dataloader in the trainer. For now just here.

            fig = display_alignment(consistent_index,
                                    self.data["gold_sure_word_alignments"][consistent_index], 
                                    self.data["gold_possible_word_alignments"][consistent_index], 
                                    self.data["word_alignments"][consistent_index], 
                                    self.data["srcs"][consistent_index].split(" "), 
                                    self.data["tgts"][consistent_index].split(" "), 
                                    dataset_name)
            # fig.savefig("temp.png")
            aggregate_metrics["plotted_consistent"] = logger.Image(fig, caption=f"step: {current_global_step}")
            plt.close()
            fig = display_alignment(consistent_index,
                                    self.data["gold_sure_token_alignments"][consistent_index], 
                                    self.data["gold_possible_token_alignments"][consistent_index], 
                                    self.data["token_alignments"][consistent_index], 
                                    self.data["token_srcs"][consistent_index].split(" "), 
                                    self.data["token_tgts"][consistent_index].split(" "), 
                                    dataset_name)
            # fig.savefig("temp.png")
            aggregate_metrics["token_plotted_consistent"] = logger.Image(fig, caption=f"step: {current_global_step}")
            plt.close()

            indices_of_importance = np.argsort(self.data["impacts"]).tolist()
            if dataset_name == "roen":
                indices_of_importance.remove(23)
            high_imact_index = indices_of_importance[-1]
            fig = display_alignment(high_imact_index,
                                    self.data["gold_sure_word_alignments"][high_imact_index], 
                                    self.data["gold_possible_word_alignments"][high_imact_index], 
                                    self.data["word_alignments"][high_imact_index], 
                                    self.data["srcs"][high_imact_index].split(" "), 
                                    self.data["tgts"][high_imact_index].split(" "), 
                                    dataset_name)
            aggregate_metrics["plotted_worst"] = logger.Image(fig, caption=f"step: {current_global_step}")
            plt.close()
            fig = display_alignment(high_imact_index,
                                    self.data["gold_sure_token_alignments"][high_imact_index], 
                                    self.data["gold_possible_token_alignments"][high_imact_index], 
                                    self.data["token_alignments"][high_imact_index], 
                                    self.data["token_srcs"][high_imact_index].split(" "), 
                                    self.data["token_tgts"][high_imact_index].split(" "), 
                                    dataset_name)
            aggregate_metrics["token_plotted_worst"] = logger.Image(fig, caption=f"step: {current_global_step}")
            plt.close()

            second_high_imact_index = indices_of_importance[-2]
            fig = display_alignment(second_high_imact_index,
                                    self.data["gold_sure_word_alignments"][second_high_imact_index],
                                    self.data["gold_possible_word_alignments"][second_high_imact_index],
                                    self.data["word_alignments"][second_high_imact_index],
                                    self.data["srcs"][second_high_imact_index].split(" "),
                                    self.data["tgts"][second_high_imact_index].split(" "), 
                                    dataset_name)
            aggregate_metrics["plotted_2nd_worst"] = logger.Image(fig, caption=f"step: {current_global_step}")
            plt.close()
            fig = display_alignment(second_high_imact_index,
                                    self.data["gold_sure_token_alignments"][second_high_imact_index], 
                                    self.data["gold_possible_token_alignments"][second_high_imact_index], 
                                    self.data["token_alignments"][second_high_imact_index], 
                                    self.data["token_srcs"][second_high_imact_index].split(" "), 
                                    self.data["token_tgts"][second_high_imact_index].split(" "), 
                                    dataset_name)
            aggregate_metrics["token_plotted_2nd_worst"] = logger.Image(fig, caption=f"step: {current_global_step}")
            plt.close()
        return aggregate_metrics
    
class BertPSIHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, 2, bias=False)

        self.bias = nn.Parameter(torch.zeros(2))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = hidden_states[:, 0]
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

from abc import ABC, abstractmethod

class BaseClassifierNet(ABC, nn.Module):
    @abstractmethod
    def get_logit_similarities(self, examples_srctgt, concat_embeddings) -> Any:
        ...
class ClassifierNet(BaseClassifierNet):
    def __init__(self, layer_norm, hidden_dim):
        super().__init__()
        self.layer_norm = layer_norm
        self.layer_norm_layer = nn.LayerNorm([768 * 2])
        self.forget_weight = nn.Sequential(
            nn.Linear(768 * 2, 1),
            nn.Sigmoid()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(768 * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 768)
        )
        self.bias = nn.Parameter(torch.tensor([0.0]))
    def embed_reps(self, x):
        if self.layer_norm:
            x_normed = self.layer_norm_layer(x)
        else:
            x_normed = x
        # residual connection with the end, and some way of forget gating the residual connection
        f_w = self.forget_weight(x_normed)
        return f_w * x[..., 768:] + (1-f_w) * self.linear_layer(x_normed) + self.bias
    def get_logit_similarities(self, examples_srctgt, concat_embeddings):
        similarities = []
        for j in range(examples_srctgt.size(0)):
            input_ids_list = examples_srctgt[j].tolist()
            sep_index = input_ids_list.index(102)
            end_index = input_ids_list.index(102, sep_index+2) # changed here to two because 102 and 101 are used to seperate sequences in awesome align.
            src_embeddings = concat_embeddings[j][1:sep_index]
            src_embeddings = self.embed_reps(src_embeddings)
            tgt_embeddings = concat_embeddings[j][sep_index+2:end_index] # also here
            tgt_embeddings = self.embed_reps(tgt_embeddings)
            similarities.append((src_embeddings[:,None,:] * tgt_embeddings[None, :, :]).sum(-1))
        return similarities
    
class ResBlock(nn.Module):
    def __init__(self, layer, hidden_size):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(hidden_size)
        # self.norm = nn.Identity()
    def forward(self, x):
        return x + self.layer(self.norm(x.permute(0,2,3,1)).permute(0,3,1,2))
    
class ConvClassifierNet(BaseClassifierNet):
    def __init__(self, hidden_dim, conv_hidden, dropout_rate):
        super().__init__()
        self.forget_weight = nn.Sequential(
            nn.Linear(768 * 2, 1),
            nn.Sigmoid()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(768 * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 768)
        )
        self.bias = torch.nn.Parameter(torch.tensor([0.0]))
        padding_mode = "zeros" # can be 'zeros', 'reflect', 'replicate' or 'circular'
        # src, tgt, dot of src tgt
        self.seq = nn.Sequential(
            nn.Conv2d(768 * 2 + 1, conv_hidden, 7, padding='same', padding_mode=padding_mode),
            nn.Sequential(nn.ReLU(), nn.Dropout2d(dropout_rate)),
            ResBlock(nn.Conv2d(conv_hidden, conv_hidden, 7, padding='same', padding_mode=padding_mode), conv_hidden),
            nn.Sequential(nn.ReLU(), nn.Dropout2d(dropout_rate)),
            ResBlock(nn.Conv2d(conv_hidden, conv_hidden, 7, padding='same', padding_mode=padding_mode), conv_hidden),
            nn.Sequential(nn.ReLU(), nn.Dropout2d(dropout_rate)),
            ResBlock(nn.Conv2d(conv_hidden, conv_hidden, 7, padding='same', padding_mode=padding_mode), conv_hidden),
            nn.Sequential(nn.ReLU(), nn.Dropout2d(dropout_rate)),
            ResBlock(nn.Conv2d(conv_hidden, conv_hidden, 7, padding='same', padding_mode=padding_mode), conv_hidden),
            nn.Sequential(nn.ReLU(), nn.Dropout2d(dropout_rate)),
            ResBlock(nn.Conv2d(conv_hidden, conv_hidden, 7, padding='same', padding_mode=padding_mode), conv_hidden),
            nn.Sequential(nn.ReLU(), nn.Dropout2d(dropout_rate)),
            ResBlock(nn.Conv2d(conv_hidden, conv_hidden, 1), conv_hidden),
            nn.ReLU(),
            nn.Conv2d(conv_hidden, 1, 1),
        )
    def get_logit_similarities(self, examples_srctgt, concat_embeddings):
        samples = []
        for j in range(examples_srctgt.size(0)):
            input_ids_list = examples_srctgt[j].tolist()
            sep_index = input_ids_list.index(102)
            end_index = input_ids_list.index(102, sep_index+2) # changed here to two because 102 and 101 are used to seperate sequences in awesome align.
            src_embeddings = concat_embeddings[j][1:sep_index]
            src_embeddings = self.embed_reps(src_embeddings)
            tgt_embeddings = concat_embeddings[j][sep_index+2:end_index] # also here
            tgt_embeddings = self.embed_reps(tgt_embeddings)
            similarity_feature = (src_embeddings[:,None,:] * tgt_embeddings[None, :, :]).sum(-1)[None, ...]
            # samples.append(torch.zeros_like(similarity_feature[0]))
            features = torch.concat((src_embeddings[:, None, :].expand(-1, tgt_embeddings.size(0), -1).permute(2,0,1), 
                                     tgt_embeddings[None, :, :].expand(src_embeddings.size(0),-1,-1).permute(2,0,1),
                                     similarity_feature), dim=0)
            similarity = self.seq(features[None,...])
            samples.append(similarity[0,0])
        return samples
    def embed_reps(self, embeddings):
        # residual connection with the end, and some way of forget gating the residual connection
        f_w = self.forget_weight(embeddings)
        return f_w * embeddings[..., 768:] + (1-f_w) * self.linear_layer(embeddings) + self.bias


class AwesomeAligner(BaseTextAligner):
    def __init__(self, 
                 maskedlm: BertForMaskedLM, 
                 classifier: BaseClassifierNet,
                 tokenizer: PreTrainedTokenizerBase, 
                 device: str, 
                 layer_number: int, 
                 threshold: float, 
                 train_supervised: bool, 
                 train_so: bool, 
                 train_psi: bool, 
                 train_mlm: bool, 
                 train_tlm: bool, 
                 train_tlm_full: bool, 
                 train_classification: bool, 
                 mlm_probability: float, 
                 supervised_weight: float,
                 so_weight: float,
                 tlm_weight: float,
                 mlm_weight: float,
                 psi_weight: float,
                 classification_weight: float,
                 word_masking: bool,
                 entropy_loss: bool,
                 div_by_len: bool,
                 cosine_sim: bool,
                 sim_func_temp: float,
                 coverage_encouragement_type: str,
                 max_softmax_temperature_start: float,
                 max_softmax_temperature_end: float,
                 coverage_weight: float,
                 classifier_threshold: float,
                 ):
        super().__init__()
        self.maskedlm = maskedlm # type: ignore # for some models you can initialize them on the device. Not bert sadly.
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.device = device
        self.layer_number = layer_number
        self.threshold = threshold
        self.train_supervised = train_supervised
        self.train_so = train_so
        self.train_psi = train_psi
        self.train_mlm = train_mlm
        self.train_tlm = train_tlm
        self.train_tlm_full = train_tlm_full
        self.train_classification = train_classification
        self.mlm_probability = mlm_probability
        self.supervised_weight = supervised_weight
        self.so_weight = so_weight
        self.tlm_weight = tlm_weight
        self.mlm_weight = mlm_weight
        self.psi_weight = psi_weight
        self.classification_weight = classification_weight
        
        # experiment vars:
        self.word_masking = word_masking
        self.entropy_loss = entropy_loss
        self.div_by_len = div_by_len
        self.cosine_sim = cosine_sim
        self.sim_func_temp = sim_func_temp
        self.coverage_encouragement_type = coverage_encouragement_type
        self.max_softmax_temperature_start = max_softmax_temperature_start
        self.max_softmax_temperature_end = max_softmax_temperature_end
        self.coverage_weight = coverage_weight
        self.classifier_threshold = classifier_threshold

        # the per encoder way to get layer hidden rep will be different. 
        # Some could possibly be achieved by the forward hook, others through what awesomealign used (rewriting the encode function to accept a layer parameter)
        self.psi_cls = BertPSIHead(self.maskedlm.bert.config.hidden_size)
        self.to(device)

    def forward(self, x):
        pass

    def training_step(self, batch: CollateFnReturn, is_val=False):
        # a dict is chosen here for our return type because most of the metrics we just want to treat uniformly, and ease of adding values is important. Type safety not so much so.
        token_alignment = None # allows for alignments to be gotten from various methods if some loss function must already produce them.
        # similarity_mat = None
        losses = dict()
        if self.train_supervised:
            supervised_loss_and_label_guesses_dict = self.get_supervised_training_loss(batch)
            supervised_loss_and_label_guesses_dict['supervised_loss_per_batch'] *= self.supervised_weight
            supervised_loss_and_label_guesses_dict['supervised_coverage_loss_per_batch'] *= self.supervised_weight
            if not is_val:
                (supervised_loss_and_label_guesses_dict['supervised_loss_per_batch'] + supervised_loss_and_label_guesses_dict['supervised_coverage_loss_per_batch']).mean().backward()
            losses["supervised_loss_per_batch"] = supervised_loss_and_label_guesses_dict['supervised_loss_per_batch'].detach()
            losses["supervised_coverage_loss_per_batch"] = supervised_loss_and_label_guesses_dict['supervised_coverage_loss_per_batch'].detach()
            if "temperature" in supervised_loss_and_label_guesses_dict:
                temp = supervised_loss_and_label_guesses_dict["temperature"]
                batch_size = batch.examples_src.size(0)
                losses["so_coverage_temp"] = torch.full((batch_size,), temp, device=self.device) # for recording
            # then we calculate prec_recall_aer_coverage (could make this optional if costs a bunch, but should be a good signal)
            token_alignment = supervised_loss_and_label_guesses_dict["token_alignment"]
            # I don't go about computing the aer prec recall for the training set during training. Could include the trianing set as one of my eval sets if I wanted this metric?        

        if self.train_classification:
            classification_loss = self.get_supervised_classification_loss(batch, is_val)
            classification_loss["classification_loss_per_batch"] *= self.classification_weight
            if not is_val:
                classification_loss["classification_loss_per_batch"].mean().backward()
            batch_size = batch.examples_srctgt.size(0)
            losses["classification_loss_per_batch"] = classification_loss["classification_loss_per_batch"].mean().detach().expand(batch_size)
            if is_val:
                # if it is val, the data is returned as a large list, this is to track the true avg nll. only approximated when training, but for val explicitly computed.
                losses["classification_loss_per_pair"] = classification_loss["classification_loss_per_batch"].flatten().tolist()
            token_alignment = classification_loss["similarity_mat"] # TODO: fix this, so it isn't as conflicting with so and normal supervised training
        if self.train_so:
            so_loss_and_label_guesses_dict = self.get_so_loss(batch)
            so_loss_and_label_guesses_dict['supervised_loss_per_batch'] *= self.so_weight
            so_loss_and_label_guesses_dict['supervised_coverage_loss_per_batch'] *= self.so_weight
            if not is_val:
                (so_loss_and_label_guesses_dict['supervised_loss_per_batch'] + so_loss_and_label_guesses_dict['supervised_coverage_loss_per_batch']).mean().backward()
            losses["so_loss_per_batch"] = so_loss_and_label_guesses_dict['supervised_loss_per_batch'].detach()
            losses["so_coverage_loss_per_batch"] = so_loss_and_label_guesses_dict['supervised_coverage_loss_per_batch'].detach()
            if "temperature" in so_loss_and_label_guesses_dict:
                temp = so_loss_and_label_guesses_dict["temperature"]
                batch_size = batch.examples_src.size(0)
                losses["so_coverage_temp"] = torch.full((batch_size,), temp, device=self.device) # for recording
            token_alignment = so_loss_and_label_guesses_dict["token_alignment"]
        
        if self.train_tlm:
            tlm_loss_dict = self.get_tlm_loss(batch, is_val)
            tlm_loss_dict["loss"] *= self.tlm_weight
            # backward is done in this function, as multiple permutations of src tgt and tgt src and maskings occur.
            batch_size = batch.examples_srctgt.size(0)
            losses["tlm_loss_per_batch"] = tlm_loss_dict["loss"].expand(batch_size) # this necessary for easier metric logging in validation loop.
        
        if self.train_mlm: # TODO: this is like 15% slower than old repo. fix that.
            mlm_loss_dict = self.get_mlm_loss(batch, is_val)
            mlm_loss_dict["loss"] *= self.mlm_weight
            batch_size = batch.examples_src.size(0)
            losses["mlm_loss_per_batch"] = mlm_loss_dict["loss"].expand(batch_size)

        if self.train_psi:
            psi_loss_dict = self.get_psi_loss(batch)
            psi_loss_dict['loss_per_batch'] *= self.psi_weight
            if not is_val:
                psi_loss_dict['loss_per_batch'].mean().backward()
            batch_size = batch.examples_src.size(0) # TODO: come up with better system than this for differently sized batches.
            losses["psi_loss_per_batch"] = psi_loss_dict["loss_per_batch"].detach().mean().expand(batch_size)
            

        if is_val and batch.alignment_construction_params["gold_possible_word_alignments"] is not None:
            if token_alignment is None:
                token_alignment, *_ = self.get_token_alignments_and_src_tgt_lens(batch) # name, *var = exp means I deconstruct a tuple, and only take the first element.
            prec_recall_aer_coverage_partial_metrics = self.get_word_prec_recall_aer_coverage_partial_metrics(
                                                            token_alignment,
                                                            batch.alignment_construction_params["bpe2word_map_src"],
                                                            batch.alignment_construction_params["bpe2word_map_tgt"],
                                                            batch.alignment_construction_params["gold_possible_word_alignments"],
                                                            batch.alignment_construction_params["gold_sure_word_alignments"])
            losses.update(prec_recall_aer_coverage_partial_metrics)
            losses.update({"srcs": batch.srcs, 
                        "tgts": batch.tgts, 
                        "gold_possible_word_alignments":batch.alignment_construction_params["gold_possible_word_alignments"], 
                        "gold_sure_word_alignments": batch.alignment_construction_params["gold_sure_word_alignments"]})
            token_srcs = []
            token_tgts = []
            gold_possible_token_alignments = []
            gold_sure_token_alignments = []
            src_lengths = []
            tgt_lengths = []
            for i in range(len(batch.srcs)):
                src, tgt = batch.srcs[i].strip().split(" "), batch.tgts[i].strip().split(" ")# [d.strip().split(" ") for d in raw_src_tgt.split(" ||| ")]
                tokenized_src_per_word = [self.tokenizer.tokenize(s) for s in src] # [['aj', '##unge'], ['!']]
                src_sub2word_mapping = sum(([j] * len(s) for j, s in enumerate(tokenized_src_per_word)), start=[])
                src_word2sub_mapping = [0] + list(accumulate(len(s) for s in tokenized_src_per_word))[:-1] # maps word to the start of it's tokenization.
                tokenized_tgt_per_word = [self.tokenizer.tokenize(t) for t in tgt]
                tgt_sub2word_mapping = sum(([j] * len(t) for j, t in enumerate(tokenized_tgt_per_word)), start=[])
                tgt_word2sub_mapping = [0] + list(accumulate(len(t) for t in tokenized_tgt_per_word))[:-1]
                possible_gold_set = batch.alignment_construction_params["gold_possible_word_alignments"][i]
                sure_gold_set = batch.alignment_construction_params["gold_sure_word_alignments"][i] # set(tuple(int(i)-1 for i in out.split('-')) for out in raw_gold.strip().split(' ') if "-" in out)
                possible_subword_gold_set = set((src_word2sub_mapping[pair[0]] + i, tgt_word2sub_mapping[pair[1]] + j) for pair in possible_gold_set for i in range(len(tokenized_src_per_word[pair[0]])) for j in range(len(tokenized_tgt_per_word[pair[1]])))
                sure_subword_gold_set = set((src_word2sub_mapping[pair[0]] + i, tgt_word2sub_mapping[pair[1]] + j) for pair in sure_gold_set for i in range(len(tokenized_src_per_word[pair[0]])) for j in range(len(tokenized_tgt_per_word[pair[1]])))
                token_srcs.append(" ".join(sum(tokenized_src_per_word, [])))
                token_tgts.append(" ".join(sum(tokenized_tgt_per_word, [])))
                gold_possible_token_alignments.append(possible_subword_gold_set)
                gold_sure_token_alignments.append(sure_subword_gold_set)
                src_lengths.append(len(src_sub2word_mapping))
                tgt_lengths.append(len(tgt_sub2word_mapping))
            token_prec_recall_aer_coverage_partial_metrics = self.get_token_prec_recall_aer_coverage_partial_metrics(
                                                            token_alignment,
                                                            gold_possible_token_alignments,
                                                            gold_sure_token_alignments,
                                                            src_lengths,
                                                            tgt_lengths)
            losses.update(token_prec_recall_aer_coverage_partial_metrics)
            losses.update({"token_srcs": token_srcs,
                        "token_tgts": token_tgts,
                        "gold_possible_token_alignments": gold_possible_token_alignments, 
                        "gold_sure_token_alignments": gold_sure_token_alignments})
            
        batch_size = batch.examples_srctgt.size(0)
        losses["loss_per_batch"] = sum((v if "loss_per_batch" in k else 0 for k, v in losses.items()), start=torch.zeros(batch_size, dtype=torch.float32, device=self.device))
        return losses

    def get_so_loss(self, batch: CollateFnReturn):
        # self-training objective loss
        with torch.no_grad(): # must be done in two steps to prevent dropout on the self guide creation.
            was_training = self.training
            self.eval()
            token_alignment, *_ = self.get_token_alignments_and_src_tgt_lens(batch)
            bpe2word_map_src = batch.alignment_construction_params["bpe2word_map_src"]
            bpe2word_map_tgt = batch.alignment_construction_params["bpe2word_map_tgt"]

            word_level_alignments = self.get_word_alignment_from_token_alignment(token_alignment, bpe2word_map_src, bpe2word_map_tgt)
            token_level_alignment_mat_self_guide = self.construct_token_level_alignment_mat_from_word_level_alignment_list(word_level_alignments, 
                                                                                                                        src_len=token_alignment.size(1), 
                                                                                                                        tgt_len=token_alignment.size(2), 
                                                                                                                        bpe2word_map_src=bpe2word_map_src, 
                                                                                                                        bpe2word_map_tgt=bpe2word_map_tgt)
            self.train(was_training)
        _, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt = self.get_token_alignments_and_src_tgt_lens(batch)
        loss_per_batch = self.get_per_batch_loss_on_token_alignments(token_level_alignment_mat_self_guide, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt, batch.step, batch.total_steps)
        loss_per_batch.update({"token_alignment": token_alignment})
        return loss_per_batch

    def get_token_alignments_and_src_tgt_lens(self, batch: CollateFnReturn):
        PAD_ID = 0
        CLS_ID = 101
        SEP_ID = 102
        negative_val = torch.finfo(torch.float32).min
        tiny_val = torch.finfo(torch.float32).tiny
        src_hidden_states = self.get_encoder_hidden_state(batch.examples_src)
        tgt_hidden_states = self.get_encoder_hidden_state(batch.examples_tgt)
        if self.cosine_sim:
            src_hidden_states_norm = src_hidden_states / (src_hidden_states.norm(dim=-1, keepdim=True) + tiny_val)
            tgt_hidden_states_norm = tgt_hidden_states / (tgt_hidden_states.norm(dim=-1, keepdim=True) + tiny_val)
            alignment_scores = (src_hidden_states_norm @ tgt_hidden_states_norm.transpose(-1,-2))
        else:
            alignment_scores = (src_hidden_states @ tgt_hidden_states.transpose(-1,-2))
        alignment_scores /= self.sim_func_temp
        src_tgt_mask = ((batch.examples_tgt == CLS_ID) | (batch.examples_tgt == SEP_ID) | (batch.examples_tgt == PAD_ID)).to(self.device)
        tgt_src_mask = ((batch.examples_src == CLS_ID) | (batch.examples_src == SEP_ID) | (batch.examples_src == PAD_ID)).to(self.device)
        len_src = (1 - tgt_src_mask.float()).sum(1)
        len_tgt = (1 - src_tgt_mask.float()).sum(1)
        src_tgt_mask = (src_tgt_mask * negative_val)[:, None, :].to(self.device)
        tgt_src_mask = (tgt_src_mask * negative_val)[:, :, None].to(self.device)
        src_tgt_masked_alignment_scores = alignment_scores + src_tgt_mask
        tgt_src_masked_alignment_scores = alignment_scores + tgt_src_mask
        # src_tgt_softmax: how the source aligns to the target (softmax across the tgt words for one src word sum to one.)
        src_tgt_softmax = torch.softmax(src_tgt_masked_alignment_scores, dim=-1)
        tgt_src_softmax = torch.softmax(tgt_src_masked_alignment_scores, dim=-2)
        token_alignment = (src_tgt_softmax > self.threshold) * (tgt_src_softmax > self.threshold)
        return token_alignment, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt

    def get_supervised_classification_forward(self, batch: CollateFnReturn):
        layer7_embeddings = self.get_encoder_hidden_state(batch.examples_srctgt)
        # print(batch.examples_srctgt[0])
        # print("look for the 102, and if there is a 101 before each 102, so I can know how to split the srctgt examples to get similarity matrices out.")
        # import ipdb; ipdb.set_trace()

        if batch.examples_srctgt.size(0) == 512:
            print("512 example src tgt hit COUNT")
        layer0_embeddings = self.maskedlm.bert.embeddings(batch.examples_srctgt)
        concat_embeddings = torch.concat([layer0_embeddings, layer7_embeddings], dim=-1)
        return self.classifier.get_logit_similarities(batch.examples_srctgt, concat_embeddings)
        # return similarities
    def get_supervised_classification_loss(self, batch: CollateFnReturn, is_val):
        token_level_alignment_mat_labels = self.construct_token_level_alignment_mat_from_word_level_alignment_list(**batch.alignment_construction_params)
        
        similarities = self.get_supervised_classification_forward(batch)
        targets = token_level_alignment_mat_labels
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            similarity_mat = torch.zeros_like(targets)
            for i, s in enumerate(similarities):
                similarity_mat[i, 1:s.shape[0]+1, 1:s.shape[1]+1] = s.sigmoid() > self.classifier_threshold
        
        targets = targets[:, 1:, 1:].split(1, dim=0) # get rid of first thing, we ideally want purely src x tgt matching.
        targets = [t[0, :s.shape[0], :s.shape[1]] for s, t in zip(similarities, targets)]
        
        similarities = torch.concat([t.flatten() for t in similarities])[None,:]
        targets = torch.concat([t.flatten() for t in targets])[None, :]
        if is_val:
            reduction = "none"
        else:
            reduction = "mean"
        loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)(similarities, targets) # [None]
        return {"classification_loss_per_batch": loss, "similarity_mat": similarity_mat.detach()}
    def get_supervised_training_loss(self, batch: CollateFnReturn):
        # (support superso... later)
        token_alignment, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt = self.get_token_alignments_and_src_tgt_lens(batch)
        token_level_alignment_mat_labels = self.construct_token_level_alignment_mat_from_word_level_alignment_list(**batch.alignment_construction_params)
        # turns out flatten.sum isn't significantly slower on mps or cpu than .sum((1,2))
        per_batch_loss = self.get_per_batch_loss_on_token_alignments(token_level_alignment_mat_labels, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt, batch.step, batch.total_steps)
        per_batch_loss.update({"token_alignment": token_alignment})
        return per_batch_loss

    def get_per_batch_loss_on_token_alignments(self, token_level_alignment_mat_labels, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt, step, total_steps):
        ret_dict = dict()
        coverage_loss = 0
        src_tgt_softmax = torch.softmax(src_tgt_masked_alignment_scores, dim=-1)
        tgt_src_softmax = torch.softmax(tgt_src_masked_alignment_scores, dim=-2)
        if self.coverage_encouragement_type == "mse_softmax":
            coverage_loss = ((src_tgt_softmax.sum(dim=-2) - 1) ** 2).flatten(1).mean(1)
            coverage_loss += ((tgt_src_softmax.sum(dim=-1) - 1) ** 2).flatten(1).mean(1)
        elif "max_softmax" in self.coverage_encouragement_type:
            if "linear" in self.coverage_encouragement_type:
                fraction_to_end = step / total_steps # expecting to get passed non zero total steps
                temp = self.max_softmax_temperature_end * fraction_to_end + self.max_softmax_temperature_start * (1-fraction_to_end)
            elif "const_cosine" in self.coverage_encouragement_type:
                fraction_to_end = max(0, -1 + 2 * step / total_steps) # expecting to get passed non zero total steps
                temp = 0.5 * (1 + np.cos(fraction_to_end * np.pi)) * (self.max_softmax_temperature_start - self.max_softmax_temperature_end) + self.max_softmax_temperature_end
            elif "cosine" in self.coverage_encouragement_type:
                fraction_to_end = step / total_steps # expecting to get passed non zero total steps
                temp = 0.5 * (1 + np.cos(fraction_to_end * np.pi)) * (self.max_softmax_temperature_start - self.max_softmax_temperature_end) + self.max_softmax_temperature_end
            else:
                raise Exception(f"need const_cosine, cosine, or linear in coverage_encouragement_type, {self.coverage_encouragement_type}")
            ret_dict.update({"temperature": temp})
            src_tgt_per_row_sum = torch.sum(src_tgt_softmax * torch.softmax((src_tgt_softmax + tgt_src_mask)/temp, dim=-2), dim=-2)
            tgt_src_per_col_sum = torch.sum(tgt_src_softmax * torch.softmax((tgt_src_softmax + src_tgt_mask)/temp, dim=-1), dim=-1)
            # if "log" in self.coverage_encouragement_type:
            #     src_tgt_per_row_sum = src_tgt_per_row_sum.log()
            #     tgt_src_per_col_sum = tgt_src_per_col_sum.log()
            coverage_loss = - src_tgt_per_row_sum.flatten(1).mean(1)
            coverage_loss -= tgt_src_per_col_sum.flatten(1).mean(1)
        # elif self.coverage_encouragement_type == "hungarian":
        #     coverage_loss = - 
        else:
            raise ValueError("must be either mse_softmax or max_softmax")
        coverage_loss_per_batch = coverage_loss * self.coverage_weight
    
        if self.entropy_loss:
            guide_src = token_level_alignment_mat_labels / (token_level_alignment_mat_labels.sum(dim=-1, keepdim=True) + 0.000001)
            guide_tgt = token_level_alignment_mat_labels / (token_level_alignment_mat_labels.sum(dim=-2, keepdim=True) + 0.000001)
            attention_log_prob_src = F.log_softmax(src_tgt_masked_alignment_scores, dim=-1)
            attention_log_prob_tgt = F.log_softmax(tgt_src_masked_alignment_scores, dim=-2)
            # had to do log_softmax for numerical reasons??? like wtf I don't really understand where the difference comes from when I manually compute kl vs with F.kl_div. I use log_softmax for both!
            so_loss_src = -F.kl_div(attention_log_prob_src, guide_src, reduction="none").flatten(1).sum(1)
            so_loss_tgt = -F.kl_div(attention_log_prob_tgt.transpose(-1,-2), guide_tgt.transpose(-1,-2), reduction="none").flatten(1).sum(1)
            # so_loss_src = torch.sum(torch.sum (torch.where(guide_src == 0, torch.zeros_like(attention_log_prob_src), attention_log_prob_src) * guide_src, -1), -1).view(-1)
            # so_loss_tgt = torch.sum(torch.sum (torch.where(guide_tgt == 0, torch.zeros_like(attention_log_prob_tgt), attention_log_prob_tgt) * guide_tgt, -1), -1).view(-1)
        else:
            so_loss_src = (token_level_alignment_mat_labels * src_tgt_softmax).flatten(1).sum(1)
            so_loss_tgt = (token_level_alignment_mat_labels * tgt_src_softmax).flatten(1).sum(1)

        if self.div_by_len:
            per_batch_loss = -so_loss_src/len_src - so_loss_tgt/len_tgt
        else:
            per_batch_loss = -so_loss_src - so_loss_tgt

        ret_dict.update({"supervised_loss_per_batch": per_batch_loss, "supervised_coverage_loss_per_batch": coverage_loss_per_batch})
        return ret_dict

    def construct_token_level_alignment_mat_from_word_level_alignment_list(self, gold_possible_word_alignments, src_len, tgt_len, bpe2word_map_src, bpe2word_map_tgt, **kwargs):
        # TODO: remove kwargs and clean up the difference between true labels and the self training labels.
        device = 'cpu' # there many small operations, better to do these on the cpu first, and then move them to gpu 10% speed up in old method, but 50% for my code because I have to create tensors.
        batch_size = len(gold_possible_word_alignments)
        token_level_alignment_mat = torch.zeros((batch_size, src_len, tgt_len), device=device)
        # I rewrote the loop here, sadly it is less readable, and only causes a 10% speedup. keeping it tho.
        for i in range(batch_size):
            gold_possible_word_alignment = gold_possible_word_alignments[i]
            bpe2word_map_src_i = torch.tensor(bpe2word_map_src[i], device=device) # ensure that the mapping isn't too large, so truncate any which would align words past 509, as cls and sep tokens are there
            bpe2word_map_tgt_i = torch.tensor(bpe2word_map_tgt[i], device=device)
            for src_index_word, tgt_index_word in gold_possible_word_alignment:
                token_level_alignment_mat[i, 1 + torch.where(bpe2word_map_src_i == src_index_word)[0][None, :, None], 1 + torch.where(bpe2word_map_tgt_i == tgt_index_word)[0][None, None, :]] = 1

        return token_level_alignment_mat.to(self.device)

    def get_encoder_hidden_state(self, input_ids, layer_number=None):
        assert isinstance(self.maskedlm, BertPreTrainedModel), "this method is specific to the BertPreTrainedModel from huggingface"
        if layer_number is None:
            layer_number = self.layer_number
        activation = dict()
        activation_name = f"layer_{layer_number}_activation"
        def hook(module, input, output):
            activation[activation_name] = output[0]
        hook_handle = self.maskedlm.bert.encoder.layer[layer_number].register_forward_hook(hook)

        # run model
        PAD_ID = 0
        attention_mask = (input_ids != PAD_ID).to(self.device)
        self.maskedlm.bert(input_ids, attention_mask=attention_mask)

        hook_handle.remove()
        return activation[activation_name]

    # def get_token_level_alignment(self, src_hidden_states, tgt_hidden_states):
    #     alignments = src_hidden_states @ tgt_hidden_states
    #     return alignments
    
    def get_psi_loss(self, batch: CollateFnReturn):
        # parallel sentence identification loss
        # for whatever reason layer_number+1 is done in awesome-align. Just replicating right now
        hidden_encoding = self.get_encoder_hidden_state(batch.psi_examples_srctgt, layer_number=self.layer_number) 

        prediction_scores_psi = self.psi_cls(hidden_encoding)
        psi_loss_per_batch = CrossEntropyLoss(reduction='none')(prediction_scores_psi.view(-1, 2), batch.psi_labels.view(-1))

        return {"loss_per_batch": psi_loss_per_batch}
        
    def get_mlm_loss(self, batch: CollateFnReturn, is_val):
        PAD_ID = 0
        

        inputs_src, labels_src, constituent_info_src = batch.input_examples_src_masked, batch.label_examples_src_masked, batch.constituent_info_from_src_masked
        inputs_tgt, labels_tgt, constituent_info_tgt = batch.input_examples_tgt_masked, batch.label_examples_tgt_masked, batch.constituent_info_from_tgt_masked
        loss_src = self.maskedlm(input_ids=inputs_src, attention_mask=(inputs_src != PAD_ID).to(self.device),  labels=labels_src, **constituent_info_src).loss
        if not is_val:
            loss_src.backward()

        loss_tgt = self.maskedlm(input_ids=inputs_tgt, attention_mask=(inputs_tgt != PAD_ID).to(self.device),  labels=labels_tgt, **constituent_info_tgt).loss
        if not is_val:
            loss_tgt.backward()
        return {"loss": loss_src.detach() + loss_tgt.detach()}
    
    def get_tlm_loss(self, batch: CollateFnReturn, is_val):
        PAD_ID = 0

        # from transformers.models.bert import BertForMaskedLM
        loss = torch.tensor(0., device=self.device)
        
        rand_ids = [0, 1]
        if not self.train_tlm_full:
            rand_ids = [int(random.random() > 0.5)]
        for rand_id in rand_ids:
            if rand_id == 0:
                iterator = zip([batch.input_examples_srctgt_mask_tgt, batch.input_examples_srctgt_mask_src],
                                [batch.label_examples_srctgt_mask_tgt, batch.label_examples_srctgt_mask_src],
                                [batch.constituent_info_from_srctgt_mask_tgt, batch.constituent_info_from_srctgt_mask_src])
            else:
                iterator = zip([batch.input_examples_tgtsrc_mask_src, batch.input_examples_tgtsrc_mask_tgt],
                                [batch.label_examples_tgtsrc_mask_src, batch.label_examples_tgtsrc_mask_tgt],
                                [batch.constituent_info_from_tgtsrc_mask_src, batch.constituent_info_from_tgtsrc_mask_tgt])
            for inputs, labels, constituent_info in iterator:
                loss_i = self.maskedlm(input_ids=inputs, attention_mask=(inputs != PAD_ID).to(self.device), labels=labels, **constituent_info).loss
                if not is_val:
                    loss_i.backward()
                loss += loss_i.detach()
        return {"loss": loss}


    def get_co_loss(self, batch: CollateFnReturn):
        # consistency optimization loss, and this will likely be combined with so_loss as done in awesome-align.
        raise NotImplementedError() 
    def compute_prec_recall_aer_coverage_partial_metrics_given_alignments(self, guess_alignments, gold_possible_alignments, gold_sure_alignments, src_lengths, tgt_lengths):
        # need to get the prec, recall, and aer metrics. This is 
        # prec: #possible correct guesses/guesses made = (|H \intersection P| / |H|)
        # recall: #sure correct guesses/possible correct = (|H \intersection S| / |S|)
        num_sure = 0
        guesses_made = 0
        guesses_made_in_possible = 0
        guesses_made_in_sure = 0

        num_elements_total = 0
        num_elements_covered = 0
        coverages = []
        impacts = []

        for i in range(len(guess_alignments)):
            guess_alignment = guess_alignments[i]
            gold_possible_alignment = gold_possible_alignments[i]
            gold_sure_alignment = gold_sure_alignments[i]

            num_sure += len(gold_sure_alignment)
            guesses_made += len(guess_alignment)
            guesses_made_in_sure += len(set(guess_alignment).intersection(gold_sure_alignment))
            guesses_made_in_possible += len(set(guess_alignment).intersection(gold_possible_alignment))
            num_elements_total = src_lengths[i] + tgt_lengths[i]
            num_elements_covered = len(set(s for s,t in guess_alignment)) + len(set(t for s,t in guess_alignment))
            coverages.append(num_elements_covered/num_elements_total)
            impacts.append((len(gold_sure_alignment) + len(guess_alignment)) - (len(set(guess_alignment).intersection(gold_sure_alignment)) + len(set(guess_alignment).intersection(gold_possible_alignment))))

        prec_recall_aer_coverage_partial_metrics = {
            "num_sure": num_sure,
            "guesses_made": guesses_made,
            "guesses_made_in_sure": guesses_made_in_sure,
            "guesses_made_in_possible": guesses_made_in_possible,
            "coverage": torch.tensor(coverages),
            "impacts": impacts,
            "alignments": guess_alignments,
        }
        return prec_recall_aer_coverage_partial_metrics
    def get_token_prec_recall_aer_coverage_partial_metrics(self, token_alignment: torch.Tensor, gold_possible_token_alignments, gold_sure_token_alignments, src_lengths, tgt_lengths):
        # get the token level alignment
        token_alignment = token_alignment.detach().cpu()
        guess_alignments = [[] for i in range(len(token_alignment))]
        for i in range(len(token_alignment)):
            guess_alignments[i] = list(zip(*(t.tolist() for t in torch.where(token_alignment[i][1:-1, 1:-1]))))

        return_dict = self.compute_prec_recall_aer_coverage_partial_metrics_given_alignments(guess_alignments, gold_possible_token_alignments, gold_sure_token_alignments, src_lengths, tgt_lengths)
        return_dict = {"token_"+key: value for key,value in return_dict.items()}
        return return_dict

    def get_word_prec_recall_aer_coverage_partial_metrics(self, token_alignment: torch.Tensor, bpe2word_map_src, bpe2word_map_tgt, gold_possible_word_alignments, gold_sure_word_alignments):
        # computes over a batch based on predictions, returns relevant information to be eventually aggragated
        # prec_recall_aer_coverage
        word_level_alignments = self.get_word_alignment_from_token_alignment(token_alignment, bpe2word_map_src, bpe2word_map_tgt)
        src_lengths = list(1 + bpe2word_map_src[i][-1] for i in range(len(bpe2word_map_src)))
        tgt_lengths = list(1 + bpe2word_map_tgt[i][-1] for i in range(len(bpe2word_map_tgt)))
        return_dict =  self.compute_prec_recall_aer_coverage_partial_metrics_given_alignments(word_level_alignments, gold_possible_word_alignments, gold_sure_word_alignments, src_lengths, tgt_lengths)
        return_dict["word_alignments"] = return_dict.pop("alignments")
        return return_dict
    def get_word_alignment_from_token_alignment(self, token_alignment: torch.Tensor, bpe2word_map_src, bpe2word_map_tgt, has_cls_in_front=True):
        # need word level alignment: using the any token matching heuristic to get at this.
        token_alignment = token_alignment.detach().cpu()
        batch_size = token_alignment.size(0)
        word_level_alignments = [set() for i in range(batch_size)]
        # could need to place this alignment matrix on cpu, as I will be decomposing it as I look for word level alignments.
        # will first try without it, and just see how the performance changes from 11.28it/s on train 30.72it/s on eval batch_size 8,  1.78 it/s on train and 3.31 it/s on eval batch_size 64.
        has_cls_in_front = 1 if has_cls_in_front else 0
        for i in range(batch_size):
            for j, k in zip(*torch.where(token_alignment[i])):
                # given that the word alignments are computed with a cls token prepended, be -1 to make alignment zero indexed.

                word_level_alignments[i].add((bpe2word_map_src[i][j - has_cls_in_front], bpe2word_map_tgt[i][k - has_cls_in_front]))
        return [list(word_level_alignment) for word_level_alignment in word_level_alignments]

    def validation_step(self, batch: CollateFnReturn):
        # both get the loss on the data, and if there are labels get the label assignment error, aer and coverage related metrics.
        losses = self.training_step(batch, is_val=True)
        return losses
    def test_step(self, batch: CollateFnReturn):
        return self.val_step(batch)
    def predict_step(self, batch: CollateFnReturn):
        pass
    # def configure_optimizers(self): 
    #     # pytorch lightning does this, but I don't get it, so will just do the init of the optimizer and scheduler inside the trainer.
    #     # update: I think I should have this now. the model is related to how it optimizes itself. It knows which parameters 
    #     #         are meant to have different learning rates and whatnot.
    #     pass
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
# aa = AwesomeAligner(BertForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased"), tokenizer, "cuda", 7, 0.001, False, False, False, False, False, False, 0.15, False, True, False, 1, "mse_softmax",1,1,0) # type: ignore
# #%%
# ids = tokenizer("hello there stranger", return_tensors='pt').input_ids.to("cuda")
# aa.mask_tokens(ids, None, None)
# #%%

from typing import Optional, Union, Tuple, List
from transformers.models.bert.modeling_bert import BertLayer, BertForMaskedLM, BertPreTrainedModel, add_start_docstrings, BERT_START_DOCSTRING, logger, add_start_docstrings_to_model_forward, BERT_INPUTS_DOCSTRING, add_code_sample_docstrings, _CHECKPOINT_FOR_DOC, MaskedLMOutput, _CONFIG_FOR_DOC, BertEmbeddings, BertPooler, BaseModelOutputWithPoolingAndCrossAttentions, _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_attention_mask_for_sdpa, BaseModelOutputWithPastAndCrossAttentions, BertOnlyMLMHead

class CustomBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.num_hidden_layers == 12, "CustomBertEncoder only works for 12 layer transformers as layers bellow 12 can be targetted for combining and expanding"
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        # add intersepting layer for decoding and uncoding the combination step
        self.combine_layer = config.combine_layer
        self.expand_layer = config.expand_layer
        self.lstm_combiner = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.combined_ffn = nn.Sequential(
            nn.Linear(in_features=config.hidden_size * 2, out_features=config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(in_features=config.hidden_size * 4, out_features=config.hidden_size)
        ) # takes cell output from the lstm combiner along with mean representation of the mask tokens before combining. The output is a residual on this mean representation.
        self.expand_ffn = nn.Sequential(
            nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(in_features=config.hidden_size * 4, out_features=config.hidden_size)
        )
        # TODO: mess around with bidirectional LSTM and using the hidden state along with the cell state?
        self.lstm_expander = nn.LSTM(config.hidden_size * 2, config.hidden_size, batch_first=True) # as input is the concatenation of the contiguous mask representation along with each of the original mask representations before the combine step.
        # expander is meant to apply as a residual to the original contextualized embedding from before the combine layer.
    
    @staticmethod
    def prepare_constituent_info_from_labels(labels, attention_mask):
        device = labels.device
        index_of_contiguous_tokens = []
        for batch_i, batch_of_labels in enumerate(labels):
            i = 0
            index_of_contiguous_tokens.append([])
            while i < len(batch_of_labels):
                if batch_of_labels[i] != -100:
                    index_of_contiguous_tokens[-1].append([i])
                    i += 1
                    while batch_of_labels[i] != -100:
                        index_of_contiguous_tokens[-1][-1].append(i)
                        i += 1
                else:
                    i += 1
        contiguous_token_lengths = torch.tensor([len(constituent_token) for constituent_tokens_in_batch in index_of_contiguous_tokens for constituent_token in constituent_tokens_in_batch], device=device)
        batch_indices_to_token_constituents = [[batch_i] for batch_i, constituent_tokens_in_batch in enumerate(index_of_contiguous_tokens) for _ in range(len(constituent_tokens_in_batch))] 
        max_contiguous_token_len = int(0 if len(contiguous_token_lengths) == 0 else contiguous_token_lengths.max().item())
        index_of_contiguous_tokens_padded = [constituent_token + [0] * (max_contiguous_token_len - len(constituent_token)) for constituent_tokens_in_batch in index_of_contiguous_tokens for constituent_token in constituent_tokens_in_batch]
        index_of_first_token_in_contiguous_tokens = [constituent_token[:1] for constituent_tokens_in_batch in index_of_contiguous_tokens for constituent_token in constituent_tokens_in_batch]
        if attention_mask is None:
            batch_size = labels.shape[0]
            seq_len = labels.shape[1]
            attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len))
        new_attention_mask_removing_contiguous_token_masks_to_one_token = attention_mask.clone()
        for batch_i in range(attention_mask.shape[0]):
            for contiguous_masks in index_of_contiguous_tokens[batch_i]:
                new_attention_mask_removing_contiguous_token_masks_to_one_token[batch_i, 0, :, contiguous_masks[1:]] = torch.finfo(torch.float32).min
        return dict(index_of_contiguous_tokens_padded=index_of_contiguous_tokens_padded,
                    contiguous_token_lengths=contiguous_token_lengths,
                    batch_indices_to_token_constituents=batch_indices_to_token_constituents,
                    index_of_first_token_in_contiguous_tokens=index_of_first_token_in_contiguous_tokens,
                    max_contiguous_token_len=max_contiguous_token_len,
                    index_of_contiguous_tokens=index_of_contiguous_tokens,
                    new_attention_mask_removing_contiguous_token_masks_to_one_token=new_attention_mask_removing_contiguous_token_masks_to_one_token)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None, # this would be used to compute where constituents are based on contiguous mask tokens, could also pass in as much info as possible
        index_of_contiguous_tokens_padded = None,
        contiguous_token_lengths = None,
        batch_indices_to_token_constituents = None,
        index_of_first_token_in_contiguous_tokens = None,
        max_contiguous_token_len = None,
        index_of_contiguous_tokens = None,
        new_attention_mask_removing_contiguous_token_masks_to_one_token = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        """
        temp area for experimenting with some input_ids and other things.
        creating here the inputs I will eventually want for computing everything I need for my master plan!!!!
        
        """
        if labels is not None:
            if index_of_contiguous_tokens_padded is None \
                or contiguous_token_lengths is None \
                or batch_indices_to_token_constituents is None \
                or index_of_first_token_in_contiguous_tokens is None \
                or max_contiguous_token_len is None \
                or index_of_contiguous_tokens is None \
                or new_attention_mask_removing_contiguous_token_masks_to_one_token is None:
                
                ret = CustomBertEncoder.prepare_constituent_info_from_labels(labels, attention_mask)
                
                index_of_contiguous_tokens_padded,\
                contiguous_token_lengths,\
                batch_indices_to_token_constituents,\
                index_of_first_token_in_contiguous_tokens,\
                max_contiguous_token_len,\
                index_of_contiguous_tokens,\
                new_attention_mask_removing_contiguous_token_masks_to_one_token = ret["index_of_contiguous_tokens_padded"], \
                        ret["contiguous_token_lengths"], \
                        ret["batch_indices_to_token_constituents"], \
                        ret["index_of_first_token_in_contiguous_tokens"], \
                        ret["max_contiguous_token_len"], \
                        ret["index_of_contiguous_tokens"], \
                        ret["new_attention_mask_removing_contiguous_token_masks_to_one_token"],
                
            old_attention_mask = attention_mask
            assert index_of_contiguous_tokens_padded is not None \
                and contiguous_token_lengths is not None \
                and batch_indices_to_token_constituents is not None \
                and index_of_first_token_in_contiguous_tokens is not None \
                and max_contiguous_token_len is not None \
                and index_of_contiguous_tokens is not None \
                and new_attention_mask_removing_contiguous_token_masks_to_one_token is not None
            
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once( # type: ignore
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        contiguous_token_reps_pre_combine_padded = None
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if i == self.combine_layer and labels is not None and len(contiguous_token_lengths) != 0: # type: ignore



                # combine the hidden_states where there are contiguous mask tokens with the combine_lstm, and use a similarly compressed attention_mask.
                # create new mask with original mask and the locations and number of contigous tokens.
                # create the combined representation, extracting out contiguous masks, and preparing them and then feeding them into the LSTM to combine. [B, L, H] 
                # lengths = [2, 1, 2]
                contiguous_token_reps_pre_combine = hidden_states[batch_indices_to_token_constituents, index_of_contiguous_tokens_padded, :]
                packed_contiguous_token_reps_pre_combine = torch.nn.utils.rnn.pack_padded_sequence(contiguous_token_reps_pre_combine, contiguous_token_lengths.cpu(), batch_first=True, enforce_sorted=False) # type: ignore
                contiguous_token_reps_pre_combine_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_contiguous_token_reps_pre_combine, batch_first=True, padding_value=0.0)
                average_contiguous_token_representation = contiguous_token_reps_pre_combine_padded.sum(-2, keepdim=True) / contiguous_token_lengths[:, None, None] # type: ignore
                every_token_output, (last_hidden, last_cell) = self.lstm_combiner(packed_contiguous_token_reps_pre_combine)
                combine_residual = self.combined_ffn(torch.concat([last_cell[0][:, None, :], average_contiguous_token_representation], dim=-1))
                contiguous_token_representation = average_contiguous_token_representation + combine_residual
                # place them into a clone of hidden_states
                hidden_states_clone = hidden_states.clone()
                hidden_states_clone[batch_indices_to_token_constituents, index_of_first_token_in_contiguous_tokens, :] = contiguous_token_representation
                hidden_states = hidden_states_clone
                attention_mask = new_attention_mask_removing_contiguous_token_masks_to_one_token # type: ignore

            if i == self.expand_layer and labels is not None and len(contiguous_token_lengths) != 0: # type: ignore

                assert contiguous_token_reps_pre_combine_padded is not None
                # expand the hidden_states and use the original mask. Use expand_lstm
                contiguous_token_representation = hidden_states[batch_indices_to_token_constituents, index_of_first_token_in_contiguous_tokens, :]
                input_to_expand_lstm = torch.concat([contiguous_token_reps_pre_combine_padded, contiguous_token_representation.repeat([1,max_contiguous_token_len,1])], dim=-1) # type: ignore
                packed_input_to_expand_lstm = torch.nn.utils.rnn.pack_padded_sequence(input_to_expand_lstm, contiguous_token_lengths.cpu(), batch_first=True, enforce_sorted=False) # type: ignore
                every_token_output, _ = self.lstm_expander(packed_input_to_expand_lstm, (torch.zeros_like(contiguous_token_representation[None, :, 0, :]), contiguous_token_representation[None, :, 0, :]))
                every_token_output, _ = torch.nn.utils.rnn.pad_packed_sequence(every_token_output, batch_first=True)
                new_mask_token_reps = contiguous_token_reps_pre_combine_padded + self.expand_ffn(every_token_output) # TODO: check that the original mask token representations are of similar magnitude at the layer I am creating
                
                hidden_states_clone = hidden_states.clone()
                j = 0
                for batch_j in range(hidden_states_clone.size(0)):
                    for constituent_token in index_of_contiguous_tokens[batch_j]: # type: ignore
                        hidden_states_clone[batch_j, constituent_token, :] = new_mask_token_reps[j][:len(constituent_token)] # type: ignore
                        j += 1
                hidden_states = hidden_states_clone
                attention_mask = old_attention_mask # type: ignore

            if self.combine_layer <= i < self.expand_layer and labels is not None and len(contiguous_token_lengths) != 0: # type: ignore
                contiguous_token_representation = hidden_states[batch_indices_to_token_constituents, index_of_first_token_in_contiguous_tokens, :]
                hidden_states_clone = hidden_states.clone()
                j = 0
                for batch_j in range(hidden_states_clone.size(0)):
                    for constituent_token in index_of_contiguous_tokens[batch_j]: # type: ignore
                        hidden_states_clone[batch_j, constituent_token, :] = contiguous_token_representation[j]
                        j += 1
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,) # type: ignore

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],) # type: ignore
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],) # type: ignore
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],) # type: ignore

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states, # type: ignore
            past_key_values=next_decoder_cache, # type: ignore
            hidden_states=all_hidden_states, # type: ignore
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class CustomBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        # self.encoder = BertEncoder(config)
        self.encoder = CustomBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value # type: ignore

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        index_of_contiguous_tokens_padded = None,
        contiguous_token_lengths = None,
        batch_indices_to_token_constituents = None,
        index_of_first_token_in_contiguous_tokens = None,
        max_contiguous_token_len = None,
        index_of_contiguous_tokens = None,
        new_attention_mask_removing_contiguous_token_masks_to_one_token = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device # type: ignore

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape) # type: ignore

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            labels=labels,
            attention_mask=extended_attention_mask,
            index_of_contiguous_tokens_padded = index_of_contiguous_tokens_padded,
            contiguous_token_lengths = contiguous_token_lengths,
            batch_indices_to_token_constituents = batch_indices_to_token_constituents,
            index_of_first_token_in_contiguous_tokens = index_of_first_token_in_contiguous_tokens,
            max_contiguous_token_len = max_contiguous_token_len,
            index_of_contiguous_tokens = index_of_contiguous_tokens,
            new_attention_mask_removing_contiguous_token_masks_to_one_token = new_attention_mask_removing_contiguous_token_masks_to_one_token,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output, # type: ignore
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class CustomBertForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]
    def __init__(self, config):
        super().__init__(config)
        # This replaces the bert portion of the model instantiated in the super().__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.bert = CustomBertModel(config, add_pooling_layer=False)

        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        index_of_contiguous_tokens_padded = None,
        contiguous_token_lengths = None,
        batch_indices_to_token_constituents = None,
        index_of_first_token_in_contiguous_tokens = None,
        max_contiguous_token_len = None,
        index_of_contiguous_tokens = None,
        new_attention_mask_removing_contiguous_token_masks_to_one_token = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            index_of_contiguous_tokens_padded = index_of_contiguous_tokens_padded,
            contiguous_token_lengths = contiguous_token_lengths,
            batch_indices_to_token_constituents = batch_indices_to_token_constituents,
            index_of_first_token_in_contiguous_tokens = index_of_first_token_in_contiguous_tokens,
            max_contiguous_token_len = max_contiguous_token_len,
            index_of_contiguous_tokens = index_of_contiguous_tokens,
            new_attention_mask_removing_contiguous_token_masks_to_one_token = new_attention_mask_removing_contiguous_token_masks_to_one_token,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1) # type: ignore
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

def customBertinit(pretrained_model_name_or_path, expand_layer, combine_layer):
    custom_bert_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    bert_base_model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    assert isinstance(bert_base_model, BertForMaskedLM) 
    custom_bert_config.combine_layer = combine_layer
    custom_bert_config.expand_layer = expand_layer
    bert_model_to_use = CustomBertForMaskedLM(custom_bert_config)
    print(bert_model_to_use.load_state_dict(bert_base_model.state_dict(), strict=False))
    # then load all the parameters into the model to avoid the issue of strange initialization.
    return bert_model_to_use



def get_collate_fn(mlm_probability, word_masking, tokenizer, block_size):
    pad_token_id = tokenizer.pad_token_id
    """ Known issue: on mac, this doesn't work with dataloaders when num_workers != 0, as the spawn processes is used to fork, and pickling a _local_ function isn't supported yet."""
    def collate_fn(examples: List[AwesomeAlignDatasetReturn]):
        examples_src, examples_tgt, examples_srctgt, examples_tgtsrc, bpe2word_map_srctgt, bpe2word_map_tgtsrc, langid_srctgt, langid_tgtsrc, psi_examples_srctgt, psi_labels = [], [], [], [], [], [], [], [], [], []
        src_len = tgt_len = 0
        bpe2word_map_src, bpe2word_map_tgt = [], []
        gold_possible_word_alignments = []
        gold_sure_word_alignments = []
        srcs = []
        tgts = []
        for example in examples:
            srcs.append(example.src)
            tgts.append(example.tgt)
            end_id = [example.src_ids[-1]]

            src_id = example.src_ids[:block_size]
            src_id = torch.LongTensor(src_id[:-1] + end_id)
            tgt_id = example.tgt_ids[:block_size]
            tgt_id = torch.LongTensor(tgt_id[:-1] + end_id)

            half_block_size = int(block_size/2)
            half_src_id = example.src_ids[:half_block_size]
            half_src_id = torch.LongTensor(half_src_id[:-1] + end_id)
            half_tgt_id = example.tgt_ids[:half_block_size]
            half_tgt_id = torch.LongTensor(half_tgt_id[:-1] + end_id)

            examples_src.append(src_id)
            examples_tgt.append(tgt_id)
            src_len = max(src_len, len(src_id))
            tgt_len = max(tgt_len, len(tgt_id))

            srctgt = torch.cat( [half_src_id, half_tgt_id] )
            langid = torch.cat([ torch.ones_like(half_src_id), torch.ones_like(half_tgt_id)*2] )
            bpe2word_map_srctgt.append([-1] + example.bpe2word_map_src[:half_block_size-2] + [-1,-1] + [i + example.bpe2word_map_src[:half_block_size-2][-1] + 1 for i in example.bpe2word_map_tgt[:half_block_size-2]] + [-1])
            examples_srctgt.append(srctgt)
            langid_srctgt.append(langid)

            tgtsrc = torch.cat( [half_tgt_id, half_src_id])
            langid = torch.cat([ torch.ones_like(half_tgt_id), torch.ones_like(half_src_id)*2] )
            examples_tgtsrc.append(tgtsrc)
            bpe2word_map_tgtsrc.append([-1] + example.bpe2word_map_tgt[:half_block_size-2] + [-1,-1] + [i + example.bpe2word_map_tgt[:half_block_size-2][-1] + 1 for i in example.bpe2word_map_src[:half_block_size-2]] + [-1])
            langid_tgtsrc.append(langid)

            # [neg, neg] pair
            neg_half_src_id = example.negative_sentence_src_ids[:half_block_size]
            neg_half_src_id = torch.LongTensor(neg_half_src_id[:-1] + end_id)
            neg_half_tgt_id = example.negative_sentence_tgt_ids[:half_block_size]
            neg_half_tgt_id = torch.LongTensor(neg_half_tgt_id[:-1] + end_id)

            if random.random()> 0.5:
                neg_srctgt = torch.cat( [neg_half_src_id, neg_half_tgt_id] )
            else:
                neg_srctgt = torch.cat( [neg_half_tgt_id, neg_half_src_id] )
            psi_examples_srctgt.append(neg_srctgt)
            psi_labels.append(1)
            # [pos, neg] pair
            rd = random.random()
            if rd > 0.75:
                neg_srctgt = torch.cat([half_src_id, neg_half_tgt_id])
            elif rd > 0.5:
                neg_srctgt = torch.cat([neg_half_src_id, half_tgt_id])
            elif rd > 0.25:
                neg_srctgt = torch.cat([half_tgt_id, neg_half_src_id])
            else:
                neg_srctgt = torch.cat([neg_half_tgt_id, half_src_id])
            psi_examples_srctgt.append(neg_srctgt)
            psi_labels.append(0)
        
            bpe2word_map_src.append(example.bpe2word_map_src[:block_size-2]) 
            bpe2word_map_tgt.append(example.bpe2word_map_tgt[:block_size-2])
            gold_possible_word_alignments.append(example.possible_labels)
            gold_sure_word_alignments.append(example.sure_labels)
        examples_src = pad_sequence(examples_src, batch_first=True, padding_value=pad_token_id)
        examples_tgt = pad_sequence(examples_tgt, batch_first=True, padding_value=pad_token_id)
        examples_srctgt = pad_sequence(examples_srctgt, batch_first=True, padding_value=pad_token_id)
        langid_srctgt = pad_sequence(langid_srctgt, batch_first=True, padding_value=pad_token_id)
        examples_tgtsrc = pad_sequence(examples_tgtsrc, batch_first=True, padding_value=pad_token_id)
        langid_tgtsrc = pad_sequence(langid_tgtsrc, batch_first=True, padding_value=pad_token_id)
        psi_examples_srctgt = pad_sequence(psi_examples_srctgt, batch_first=True, padding_value=pad_token_id)
        psi_labels = torch.tensor(psi_labels)
        if gold_possible_word_alignments[0] is None:
            gold_possible_word_alignments = None
        device = "cpu"
        # TODO: fix the langid_mask so that I just return all possible options that could be run on... I wonder if this will even speed it up...
        def mask_tokens(inputs: torch.Tensor, bpe2word_map, langid_mask=None, lang_id=None, recurse=True) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Taken from Awesome-align directly, with minimal modification.
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            MASK_ID = 103
            if tokenizer.mask_token is None:
                raise ValueError(
                    "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
                )
            mask_type = torch.bool

            labels = inputs.detach().clone()
            inputs = inputs.detach().clone()

            if word_masking and recurse:
                # if -1 is present in the map, that means it is a special token, and should be ignored for masking purposes...
                new_inputs = []
                new_labels = []
                for i in range(inputs.shape[0]):
                    # perform forward mapping from tokens to words for word level masking, random, or identity transforms
                    visited = set()
                    word_input_i_list = []
                    special_tokens = 0
                    bpe2word_map_i = torch.full_like(inputs[i], fill_value=-1)
                    bpe2word_map_i[:len(bpe2word_map[i])] = torch.tensor(bpe2word_map[i], device=device)
                    for token in bpe2word_map[i]:
                        if token == -1:
                            word_input_i_list.append(101 if special_tokens % 2 == 0 else 102)
                            special_tokens += 1
                        elif token not in visited:
                            word_input_i_list.append(110)
                            visited.add(token)
                    assert special_tokens % 2 == 0 , "there should be 2 or 4 total special tokens by the end"
                    word_input_i = torch.tensor(word_input_i_list, device=device)

                    if langid_mask is not None:
                        first_id = langid_mask[i][0]
                        last_id = 3 - first_id # either 1 or 2.
                        word_langid_mask_i = torch.full_like(word_input_i, fill_value=last_id)
                        word_langid_mask_i[:word_input_i_list[1:].index(101) + 1] = first_id
                        word_langid_mask_i = word_langid_mask_i[None]
                    else:
                        word_langid_mask_i = None
                    new_word_input_i, new_word_label_i = mask_tokens(word_input_i[None], None, word_langid_mask_i, lang_id, recurse=False)
                    
                    # perform reverse mapping from words to tokens
                    new_input_i = inputs[i].detach().clone()
                    new_label_i = torch.full_like(inputs[i], -100)
                    for j in range(len(word_input_i_list)):
                        word_number = j - (1 if j < word_input_i_list.index(102) else 3)
                        if new_word_input_i[0][j] == new_word_label_i[0][j]:
                            # keep the same:
                            new_label_i[bpe2word_map_i == word_number] = new_input_i[bpe2word_map_i == word_number]
                        elif new_word_input_i[0][j] == MASK_ID:
                            # masked indices
                            new_label_i[bpe2word_map_i == word_number] = new_input_i[bpe2word_map_i == word_number]
                            new_input_i[bpe2word_map_i == word_number] = MASK_ID
                        elif (new_word_input_i[0][j] != new_word_label_i[0][j]) and (new_word_label_i[0][j] != -100):
                            # changed indices
                            new_label_i[bpe2word_map_i == word_number] = new_input_i[bpe2word_map_i == word_number]
                            new_input_i[bpe2word_map_i == word_number] = torch.randint(len(tokenizer), new_input_i[bpe2word_map_i == word_number].shape, dtype=torch.long, device=device)
                    
                    new_inputs.append(new_input_i)
                    new_labels.append(new_label_i)
                return torch.stack(new_inputs), torch.stack(new_labels)
            # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
            probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=mask_type, device=device), value=0.0)
            if tokenizer._pad_token is not None:
                padding_mask = labels.eq(tokenizer.pad_token_id) # type: ignore
                probability_matrix.masked_fill_(padding_mask, value=0.0)

            if langid_mask is not None:
                padding_mask = langid_mask.eq(lang_id)
                probability_matrix.masked_fill_(padding_mask, value=0.0)

            masked_indices = torch.bernoulli(probability_matrix).to(mask_type)
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).to(mask_type) & masked_indices
            inputs[indices_replaced] = MASK_ID

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).to(mask_type) & masked_indices & ~indices_replaced
            random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

        
        input_examples_srctgt_mask_tgt, label_examples_srctgt_mask_tgt = mask_tokens(examples_srctgt, bpe2word_map_srctgt, langid_srctgt, 1)
        input_examples_srctgt_mask_src, label_examples_srctgt_mask_src = mask_tokens(examples_srctgt, bpe2word_map_srctgt, langid_srctgt, 2)
        input_examples_tgtsrc_mask_src, label_examples_tgtsrc_mask_src = mask_tokens(examples_tgtsrc, bpe2word_map_tgtsrc, langid_tgtsrc, 1)
        input_examples_tgtsrc_mask_tgt, label_examples_tgtsrc_mask_tgt = mask_tokens(examples_tgtsrc, bpe2word_map_tgtsrc, langid_tgtsrc, 2)
        input_examples_src_masked, label_examples_src_masked = mask_tokens(examples_src, [[-1] + l + [-1] for l in bpe2word_map_src])
        input_examples_tgt_masked, label_examples_tgt_masked = mask_tokens(examples_tgt, [[-1] + l + [-1] for l in bpe2word_map_tgt])

        constituent_info_from_srctgt_mask_tgt = CustomBertEncoder.prepare_constituent_info_from_labels(label_examples_srctgt_mask_tgt, _prepare_4d_attention_mask_for_sdpa(input_examples_srctgt_mask_tgt != pad_token_id, dtype=torch.float32))
        constituent_info_from_srctgt_mask_src = CustomBertEncoder.prepare_constituent_info_from_labels(label_examples_srctgt_mask_src, _prepare_4d_attention_mask_for_sdpa(input_examples_srctgt_mask_src != pad_token_id, dtype=torch.float32))
        constituent_info_from_tgtsrc_mask_src = CustomBertEncoder.prepare_constituent_info_from_labels(label_examples_tgtsrc_mask_src, _prepare_4d_attention_mask_for_sdpa(input_examples_tgtsrc_mask_src != pad_token_id, dtype=torch.float32))
        constituent_info_from_tgtsrc_mask_tgt = CustomBertEncoder.prepare_constituent_info_from_labels(label_examples_tgtsrc_mask_tgt, _prepare_4d_attention_mask_for_sdpa(input_examples_tgtsrc_mask_tgt != pad_token_id, dtype=torch.float32))
        constituent_info_from_src_masked = CustomBertEncoder.prepare_constituent_info_from_labels(label_examples_src_masked, _prepare_4d_attention_mask_for_sdpa(input_examples_src_masked != pad_token_id, dtype=torch.float32))
        constituent_info_from_tgt_masked = CustomBertEncoder.prepare_constituent_info_from_labels(label_examples_tgt_masked, _prepare_4d_attention_mask_for_sdpa(input_examples_tgt_masked != pad_token_id, dtype=torch.float32))


        # at some point I have to create the alignments between the source and the target through the model itself, and I have the necessary parameters here for doing that, so I should just give what parameters I have to the function which is in charege of that
        alignment_construction_params = dict(inputs_src=examples_src, inputs_tgt=examples_tgt, bpe2word_map_src=bpe2word_map_src, bpe2word_map_tgt=bpe2word_map_tgt, src_len=src_len, tgt_len=tgt_len, gold_possible_word_alignments=gold_possible_word_alignments, gold_sure_word_alignments=gold_sure_word_alignments)
        # need to add new things to the collate function to make the model faster. This is the masking of words should be done in the collate function rather than in the model.
        # also, should include things which are currently computed in the model like the 
        # index_of_contiguous_tokens_padded
        # contiguous_token_lengths
        # batch_indices_to_token_constituents
        # index_of_first_token_in_contiguous_tokens
        # max_contiguous_token_len
        # index_of_contiguous_tokens
        # new_attention_mask_removing_contiguous_token_masks_to_one_token
        return CollateFnReturn(
            srcs = srcs,
            tgts = tgts,
            examples_src = examples_src, 
            examples_tgt = examples_tgt,
            alignment_construction_params = alignment_construction_params,
            bpe2word_map_src=[[-1] + l + [-1] for l in bpe2word_map_src],
            bpe2word_map_tgt=[[-1] + l + [-1] for l in bpe2word_map_tgt],
            examples_srctgt = examples_srctgt,
            bpe2word_map_srctgt=bpe2word_map_srctgt,
            langid_srctgt = langid_srctgt, 
            input_examples_src_masked=input_examples_src_masked,
            label_examples_src_masked=label_examples_src_masked,
            input_examples_tgt_masked=input_examples_tgt_masked,
            label_examples_tgt_masked=label_examples_tgt_masked,
            input_examples_srctgt_mask_tgt=input_examples_srctgt_mask_tgt, 
            label_examples_srctgt_mask_tgt=label_examples_srctgt_mask_tgt,
            input_examples_srctgt_mask_src=input_examples_srctgt_mask_src, 
            label_examples_srctgt_mask_src=label_examples_srctgt_mask_src,
            input_examples_tgtsrc_mask_src=input_examples_tgtsrc_mask_src, 
            label_examples_tgtsrc_mask_src=label_examples_tgtsrc_mask_src,
            input_examples_tgtsrc_mask_tgt=input_examples_tgtsrc_mask_tgt, 
            label_examples_tgtsrc_mask_tgt=label_examples_tgtsrc_mask_tgt,
            constituent_info_from_srctgt_mask_tgt=constituent_info_from_srctgt_mask_tgt,
            constituent_info_from_srctgt_mask_src=constituent_info_from_srctgt_mask_src,
            constituent_info_from_tgtsrc_mask_src=constituent_info_from_tgtsrc_mask_src,
            constituent_info_from_tgtsrc_mask_tgt=constituent_info_from_tgtsrc_mask_tgt,
            constituent_info_from_src_masked=constituent_info_from_src_masked,
            constituent_info_from_tgt_masked=constituent_info_from_tgt_masked,
            examples_tgtsrc = examples_tgtsrc, 
            bpe2word_map_tgtsrc=bpe2word_map_tgtsrc,
            langid_tgtsrc = langid_tgtsrc, 
            psi_examples_srctgt = psi_examples_srctgt, 
            psi_labels = psi_labels,
            step=0, 
            total_steps=0
        )
    return collate_fn

if __name__ == "__main__":
    pass

