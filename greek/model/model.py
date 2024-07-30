from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
import random
from torch.nn.utils.rnn import pad_sequence
from greek.dataset.dataset import AwesomeAlignDatasetReturn
from collections import UserDict
# collate function is deeply related to how the train function works, so I will define them in the same place.
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers import PreTrainedTokenizerBase



@dataclass
class CollateFnReturn:
    examples_src: torch.Tensor
    examples_tgt: torch.Tensor
    alignment_construction_params: Dict[str, Any]
    examples_srctgt: torch.Tensor
    langid_srctgt: torch.Tensor
    examples_tgtsrc: torch.Tensor
    langid_tgtsrc: torch.Tensor
    psi_examples_srctgt: torch.Tensor
    psi_labels: torch.Tensor
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
    def update(self, other: Dict):
        if len(self.data) != 0:
            assert other.keys() == self.data.keys(), f"must have the same keys to be able to update the values in the metric item, but got {list(other.keys())=} instead of {list(self.data.keys())}"
        # average by default, and then for specific keys, do specific things.

        num_elements = 0
        for key, value in other.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu() # .mean().item()
                if num_elements == 0:
                    num_elements = value.size(0)
                else:
                    assert num_elements == value.size(0), "we only allow tensors of batch size for ease of determining the number of elements"
                value = value.sum().item()

            self.data[key] = self.data.get(key, 0) + value # the default is in the case that this is the first update call.
        self.total_num_elements += num_elements

    def compute(self):
        """computes the aggigate metrics for this class to be returned for logging/printing."""
        aggregate_metrics = dict()

        for key in self.data.keys():
            if "loss_per_batch" in key:
                aggregate_metrics["avg_" + key.removesuffix("_per_batch")] = self.data[key] / self.total_num_elements
            elif key not in ["guesses_made_in_possible", "guesses_made", "num_sure", "guesses_made_in_sure"]:
                aggregate_metrics["avg_" + key] = self.data[key] / self.total_num_elements
        aggregate_metrics["total_num_elements"] = self.total_num_elements

        aggregate_metrics["Precision"] = self.data["guesses_made_in_possible"] / self.data["guesses_made"]
        aggregate_metrics["Recall"] = self.data["guesses_made_in_sure"] / self.data["num_sure"]
        aggregate_metrics["AER"] = 1 - ((self.data["guesses_made_in_possible"] + self.data["guesses_made_in_sure"]) / (self.data["guesses_made"] + self.data["num_sure"]))


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

        
class AwesomeAligner(BaseTextAligner):
    def __init__(self, 
                 maskedlm: BertForMaskedLM, 
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
                 mlm_probability: float, 
                 entropy_loss: bool,
                 div_by_len: bool,
                 cosine_sim: bool,
                 sim_func_temp: float,
                 coverage_encouragement_type: str,
                 max_softmax_temperature: float,
                 coverage_weight: float
                 ):
        super().__init__()
        self.maskedlm = maskedlm # type: ignore # for some models you can initialize them on the device. Not bert sadly.
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
        self.mlm_probability = mlm_probability
        
        # experiment vars:
        self.entropy_loss = entropy_loss
        self.div_by_len = div_by_len
        self.cosine_sim = cosine_sim
        self.sim_func_temp = sim_func_temp
        self.coverage_encouragement_type = coverage_encouragement_type
        self.max_softmax_temperature = max_softmax_temperature
        self.coverage_weight = coverage_weight

        # the per encoder way to get layer hidden rep will be different. 
        # Some could possibly be achieved by the forward hook, others through what awesomealign used (rewriting the encode function to accept a layer parameter)
        self.psi_cls = BertPSIHead(self.maskedlm.bert.config.hidden_size)
        self.to(device)

    def forward(self, x):
        pass

    def training_step(self, batch: CollateFnReturn, is_val=False):
        # a dict is chosen here for our return type because most of the metrics we just want to treat uniformly, and ease of adding values is important. Type safety not so much so.
        token_alignment = None # allows for alignments to be gotten from various methods if some loss function must already produce them.
        losses = dict()
        if self.train_supervised:
            supervised_loss_and_label_guesses_dict = self.get_supervised_training_loss(batch)
            if not is_val:
                supervised_loss_and_label_guesses_dict['loss_per_batch'].mean().backward()
            losses["supervised_loss_per_batch"] = supervised_loss_and_label_guesses_dict['loss_per_batch'].detach()
            # then we calculate prec_recall_aer_coverage (could make this optional if costs a bunch, but should be a good signal)
            token_alignment = supervised_loss_and_label_guesses_dict["token_alignment"]

            prec_recall_aer_coverage_partial_metrics = self.get_word_prec_recall_aer_coverage_partial_metrics(
                                                            token_alignment,
                                                            batch.alignment_construction_params["bpe2word_map_src"],
                                                            batch.alignment_construction_params["bpe2word_map_tgt"],
                                                            batch.alignment_construction_params["gold_possible_word_alignments"],
                                                            batch.alignment_construction_params["gold_sure_word_alignments"],
                                                            )
            losses.update(prec_recall_aer_coverage_partial_metrics)
        
        if self.train_so:
            so_loss_and_label_guesses_dict = self.get_so_loss(batch)
            if not is_val:
                so_loss_and_label_guesses_dict['loss_per_batch'].mean().backward()
            losses["so_loss_per_batch"] = so_loss_and_label_guesses_dict['loss_per_batch'].detach()
            token_alignment = so_loss_and_label_guesses_dict["token_alignment"]
        
        if self.train_tlm:
            tlm_loss_dict = self.get_tlm_loss(batch, is_val)
            # backward is done in this function, as multiple permutations of src tgt and tgt src and maskings occur.
            batch_size = batch.examples_srctgt.size(0)
            losses["tlm_loss_per_batch"] = tlm_loss_dict["loss"].expand(batch_size) # this necessary for easier metric logging in validation loop.
        
        if self.train_mlm: # TODO: this is like 15% slower than old repo. fix that.
            mlm_loss_dict = self.get_mlm_loss(batch, is_val)
            batch_size = batch.examples_src.size(0)
            losses["mlm_loss_per_batch"] = mlm_loss_dict["loss"].expand(batch_size)

        if self.train_psi:
            psi_loss_dict = self.get_psi_loss(batch)
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
                                                            batch.alignment_construction_params["gold_sure_word_alignments"],
                                                            )
            losses.update(prec_recall_aer_coverage_partial_metrics)
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
        loss_per_batch = self.get_per_batch_loss_on_token_alignments(token_level_alignment_mat_self_guide, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt)
        return {"loss_per_batch": loss_per_batch, "token_alignment": token_alignment}

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

    def get_supervised_training_loss(self, batch: CollateFnReturn):
        # (support superso... later)
        token_alignment, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt = self.get_token_alignments_and_src_tgt_lens(batch)
        token_level_alignment_mat_labels = self.construct_token_level_alignment_mat_from_word_level_alignment_list(**batch.alignment_construction_params)
        # turns out flatten.sum isn't significantly slower on mps or cpu than .sum((1,2))
        per_batch_loss = self.get_per_batch_loss_on_token_alignments(token_level_alignment_mat_labels, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt)
        return {"loss_per_batch": per_batch_loss, "token_alignment": token_alignment}

    def get_per_batch_loss_on_token_alignments(self, token_level_alignment_mat_labels, src_tgt_masked_alignment_scores, tgt_src_masked_alignment_scores, src_tgt_mask, tgt_src_mask, len_src, len_tgt):
        
        # TODO: fix softmax with masking for max_softmax
        #       I screwed up the fact that masking heavily effects the conceptual interpretation of coverage's implementation
        coverage_loss = 0
        src_tgt_softmax = torch.softmax(src_tgt_masked_alignment_scores, dim=-1)
        tgt_src_softmax = torch.softmax(tgt_src_masked_alignment_scores, dim=-2)
        if self.coverage_encouragement_type == "mse_softmax":
            coverage_loss = ((src_tgt_softmax.sum(dim=-2) - 1) ** 2).flatten(1).mean(1)
            coverage_loss += ((tgt_src_softmax.sum(dim=-1) - 1) ** 2).flatten(1).mean(1)
        elif self.coverage_encouragement_type == "max_softmax":
            coverage_loss = - torch.sum(src_tgt_softmax * torch.softmax((src_tgt_softmax + tgt_src_mask)/self.max_softmax_temperature, dim=-2), dim=-2).flatten(1).mean(1)
            coverage_loss -= torch.sum(tgt_src_softmax * torch.softmax((tgt_src_softmax + src_tgt_mask)/self.max_softmax_temperature, dim=-1), dim=-1).flatten(1).mean(1)
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

        return per_batch_loss + coverage_loss_per_batch

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
        assert isinstance(self.maskedlm, BertForMaskedLM), "this method is specific to the BERTForMaskedLM from huggingface"
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
        hidden_encoding = self.get_encoder_hidden_state(batch.psi_examples_srctgt, layer_number=self.layer_number+1) 

        prediction_scores_psi = self.psi_cls(hidden_encoding)
        psi_loss_per_batch = CrossEntropyLoss(reduction='none')(prediction_scores_psi.view(-1, 2), batch.psi_labels.view(-1))

        return {"loss_per_batch": psi_loss_per_batch}
        
    def get_mlm_loss(self, batch: CollateFnReturn, is_val):
        PAD_ID = 0
        src_tokens = batch.examples_src
        tgt_tokens = batch.examples_src
        inputs_src, labels_src = self.mask_tokens(src_tokens)
        inputs_tgt, labels_tgt = self.mask_tokens(tgt_tokens)
        loss_src = self.maskedlm(input_ids=inputs_src, attention_mask=(inputs_src != PAD_ID).to(self.device),  labels=labels_src).loss
        if not is_val:
            loss_src.backward()

        loss_tgt = self.maskedlm(input_ids=inputs_tgt, attention_mask=(inputs_tgt != PAD_ID).to(self.device),  labels=labels_tgt).loss
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
                select_srctgt = batch.examples_srctgt
                select_langid = batch.langid_srctgt
            else:
                select_srctgt = batch.examples_tgtsrc
                select_langid = batch.langid_tgtsrc
            for lang_id in [1, 2]:
                with torch.no_grad():
                    inputs_srctgt, labels_srctgt = self.mask_tokens(select_srctgt, select_langid, lang_id)
                loss_i = self.maskedlm(input_ids=inputs_srctgt, attention_mask=(inputs_srctgt != PAD_ID).to(self.device), labels=labels_srctgt).loss
                if not is_val:
                    loss_i.backward()
                loss += loss_i.detach()
        return {"loss": loss}

    def mask_tokens(self, inputs: torch.Tensor, langid_mask=None, lang_id=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        MASK_ID = 103
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        mask_type = torch.bool

        labels = inputs.detach().clone()
        inputs = inputs.detach().clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=self.device)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=mask_type, device=self.device), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id) # type: ignore
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        if langid_mask is not None:
            padding_mask = langid_mask.eq(lang_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
            
        masked_indices = torch.bernoulli(probability_matrix).to(mask_type)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=self.device)).to(mask_type) & masked_indices
        inputs[indices_replaced] = MASK_ID

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=self.device)).to(mask_type) & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=self.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


    def get_co_loss(self, batch: CollateFnReturn):
        # consistency optimization loss, and this will likely be combined with so_loss as done in awesome-align.
        raise NotImplementedError() 
    
    def get_bpe_prec_recall_aer_coverage(self,):
        # Should I include this? our hope is to do one to one matching and just add word, phrase, and sentence level encodings, so this won't matter...
        # I will delay writing this. 
        raise NotImplementedError() 

    def get_word_prec_recall_aer_coverage_partial_metrics(self, token_alignment: torch.Tensor, bpe2word_map_src, bpe2word_map_tgt, gold_possible_word_alignments, gold_sure_word_alignments):
        # computes over a batch based on predictions, returns relevant information to be eventually aggragated
        # prec_recall_aer_coverage
        # need to get the prec, recall, and aer metrics. This is 
        # prec: #possible correct guesses/guesses made = (|H \intersection P| / |H|)
        # recall: #sure correct guesses/possible correct = (|H \intersection S| / |S|)

        # get the token level alignment
        batch_size = token_alignment.size(0)
        word_level_alignments = self.get_word_alignment_from_token_alignment(token_alignment, bpe2word_map_src, bpe2word_map_tgt)

        num_sure = 0
        guesses_made = 0
        guesses_made_in_possible = 0
        guesses_made_in_sure = 0

        num_words_total = 0
        num_words_covered = 0
        coverage = []
        for i in range(batch_size):
            word_level_alignment = word_level_alignments[i]
            gold_possible_word_alignment = gold_possible_word_alignments[i]
            gold_sure_word_alignment = gold_sure_word_alignments[i]

            num_sure += len(gold_sure_word_alignment)
            guesses_made += len(word_level_alignment)
            guesses_made_in_sure += len(set(word_level_alignment).intersection(gold_sure_word_alignment))
            guesses_made_in_possible += len(set(word_level_alignment).intersection(gold_possible_word_alignment))
            num_words_total = (1 + bpe2word_map_src[i][-1]) + (1 + bpe2word_map_tgt[i][-1]) # this doesn't account for the fact we only take the first 510 tokens to align.
            num_words_covered = len(set(s for s,t in word_level_alignment)) + len(set(t for s,t in word_level_alignment))
            coverage.append(num_words_covered/num_words_total)

        prec_recall_aer_coverage_partial_metrics = {
            "num_sure": num_sure,
            "guesses_made": guesses_made,
            "guesses_made_in_sure": guesses_made_in_sure,
            "guesses_made_in_possible": guesses_made_in_possible,
            "coverage": torch.tensor(coverage),
        }
        return prec_recall_aer_coverage_partial_metrics

    def get_word_alignment_from_token_alignment(self, token_alignment: torch.Tensor, bpe2word_map_src, bpe2word_map_tgt):
        # need word level alignment: using the any token matching heuristic to get at this.
        token_alignment = token_alignment.detach().cpu()
        batch_size = token_alignment.size(0)
        word_level_alignments = [set() for i in range(batch_size)]
        # could need to place this alignment matrix on cpu, as I will be decomposing it as I look for word level alignments.
        # will first try without it, and just see how the performance changes from 11.28it/s on train 30.72it/s on eval batch_size 8,  1.78 it/s on train and 3.31 it/s on eval batch_size 64.
        
        for i in range(batch_size):
            for j, k in zip(*torch.where(token_alignment[i])):
                # given that the word alignments are computed with a cls token prepended, be -1 to make alignment zero indexed.
                word_level_alignments[i].add((bpe2word_map_src[i][j - 1], bpe2word_map_tgt[i][k - 1]))
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


def get_collate_fn(pad_token_id, block_size):
    """ Known issue: on mac, this doesn't work with dataloaders when num_workers != 0, as the spawn processes is used to fork, and pickling a _local_ function isn't supported yet."""
    def collate_fn(examples: List[AwesomeAlignDatasetReturn]):
        examples_src, examples_tgt, examples_srctgt, examples_tgtsrc, langid_srctgt, langid_tgtsrc, psi_examples_srctgt, psi_labels = [], [], [], [], [], [], [], []
        src_len = tgt_len = 0
        bpe2word_map_src, bpe2word_map_tgt = [], []
        gold_possible_word_alignments = []
        gold_sure_word_alignments = []
        for example in examples:
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
            examples_srctgt.append(srctgt)
            langid_srctgt.append(langid)

            tgtsrc = torch.cat( [half_tgt_id, half_src_id])
            langid = torch.cat([ torch.ones_like(half_tgt_id), torch.ones_like(half_src_id)*2] )
            examples_tgtsrc.append(tgtsrc)
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
        
            # block_size-2 allows the implementation of construct_token_level_alignment_mat_from_word_level_alignment_list 
            # not to have to know the block_size, as some words are cut off in long encoding situation, and therefore we 
            # cant use all potential word alignments without more complex model encoding, which isn't worth it at this stage 
            # in development. There are easier gains to be made, and the primary use case will eventually be for extra long 
            # contexts, but not yet the focus.
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

        # at some point I have to create the alignments between the source and the target through the model itself, and I have the necessary parameters here for doing that, so I should just give what parameters I have to the function which is in charege of that
        alignment_construction_params = dict(inputs_src=examples_src, inputs_tgt=examples_tgt, bpe2word_map_src=bpe2word_map_src, bpe2word_map_tgt=bpe2word_map_tgt, src_len=src_len, tgt_len=tgt_len, gold_possible_word_alignments=gold_possible_word_alignments, gold_sure_word_alignments=gold_sure_word_alignments)
        # TODO: there could be a difference between using this and supervised finetuning, might want to get the loss from SO as a possible metric in eval seperate from AER
        #   then could add new loss called supervised loss which requires the label to be given, and allows for recording of this loss
        # return [examples_src, examples_tgt, alignment_construction_params, examples_srctgt, langid_srctgt, examples_tgtsrc, langid_tgtsrc, psi_examples_srctgt, psi_labels]
        batched_examples = CollateFnReturn(
            examples_src = examples_src, 
            examples_tgt = examples_tgt,
            alignment_construction_params = alignment_construction_params,
            examples_srctgt = examples_srctgt, 
            langid_srctgt = langid_srctgt, 
            examples_tgtsrc = examples_tgtsrc, 
            langid_tgtsrc = langid_tgtsrc, 
            psi_examples_srctgt = psi_examples_srctgt, 
            psi_labels = psi_labels, 
        )
        return batched_examples
    return collate_fn

if __name__ == "__main__":
    pass
