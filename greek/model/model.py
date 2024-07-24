from abc import abstractmethod
import torch.nn as nn
import torch
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
import random
from torch.nn.utils.rnn import pad_sequence
from greek.dataset.dataset import AwesomeAlignDatasetReturn
from collections import UserDict
# collate function is deeply related to how the train function works, so I will define them in the same place.


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

@dataclass
class AwesomeAlignerValReturn:
    total_num_elements: int = 0
    data: Dict = field(default_factory=dict)
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
            if key not in ["loss_per_batch", "guesses_made_in_possible", "guesses_made", "num_sure", "guesses_made_in_sure"]:
                aggregate_metrics["avg_" + key] = self.data[key] / self.total_num_elements

        aggregate_metrics["avg_loss"] = self.data["loss_per_batch"] / self.total_num_elements
        aggregate_metrics["total_num_elements"] = self.total_num_elements
        
        aggregate_metrics["Precision"] = self.data["guesses_made_in_possible"] / self.data["guesses_made"]
        aggregate_metrics["Recall"] = self.data["guesses_made_in_sure"] / self.data["num_sure"]
        aggregate_metrics["AER"] = 1 - ((self.data["guesses_made_in_possible"] + self.data["guesses_made_in_sure"]) / (self.data["guesses_made"] + self.data["num_sure"]))

        return aggregate_metrics

        
class AwesomeAligner(BaseTextAligner):
    def __init__(self, maskedlm: nn.Module, device: str, layer_number: int, threshold: float):
        super().__init__()
        self.maskedlm = maskedlm.to(device) # for some models you can initialize them on the device. Not bert sadly.
        self.layer_number = layer_number
        self.threshold = threshold
        self.device = device
        # the per encoder way to get layer hidden rep will be different. 
        # Some could possibly be achieved by the forward hook, others through what awesomealign used (rewriting the encode function to accept a layer parameter)
    
    def forward(self, x):
        pass


    def training_step(self, batch: CollateFnReturn):
        # a dict is chosen here for our return type because most of the metrics we just want to treat uniformly, and ease of adding values is important. Type safety not so much so.

        losses = dict()
        supervised_loss_and_label_guesses_dict = self.get_supervised_training_loss(batch)
        losses["loss_per_batch"] = supervised_loss_and_label_guesses_dict['loss_per_batch']
        # then we calculate prec_recall_aer_coverage (could make this optional if costs a bunch, but should be a good signal)
        prec_recall_aer_coverage_partial_metrics = self.get_word_prec_recall_aer_coverage_partial_metrics(supervised_loss_and_label_guesses_dict["src_tgt_softmax"],
                                                               supervised_loss_and_label_guesses_dict["tgt_src_softmax"],
                                                               batch.alignment_construction_params["bpe2word_map_src"],
                                                               batch.alignment_construction_params["bpe2word_map_tgt"],
                                                               batch.alignment_construction_params["gold_possible_word_alignments"],
                                                               batch.alignment_construction_params["gold_sure_word_alignments"],
                                                               )
        losses.update(prec_recall_aer_coverage_partial_metrics)
        return losses

    def get_overall_loss(self, batch: CollateFnReturn):
        raise NotImplementedError()

    def get_so_loss(self, batch: CollateFnReturn):
        # self-training objective loss
        pass


    def get_supervised_training_loss(self, batch: CollateFnReturn):
        # get labels from the word level alignments passed in.
        # get the alignments from the underlying encoder model,
        # (support superso... later)
        PAD_ID = 0
        CLS_ID = 101
        SEP_ID = 102
        src_hidden_states = self.get_encoder_hidden_state(batch.examples_src)
        tgt_hidden_states = self.get_encoder_hidden_state(batch.examples_tgt)
        alignment_scores = (src_hidden_states @ tgt_hidden_states.transpose(-1,-2))
        src_tgt_mask = ((batch.examples_tgt == CLS_ID) | (batch.examples_tgt == SEP_ID) | (batch.examples_tgt == PAD_ID)).to(self.device)
        tgt_src_mask = ((batch.examples_src == CLS_ID) | (batch.examples_src == SEP_ID) | (batch.examples_src == PAD_ID)).to(self.device)
        len_src = (1 - src_tgt_mask.float()).sum(1)
        len_tgt = (1 - tgt_src_mask.float()).sum(1)
        src_tgt_mask = (src_tgt_mask * torch.finfo(torch.float32).min)[:, None, :].to(self.device)
        tgt_src_mask = (tgt_src_mask * torch.finfo(torch.float32).min)[:, :, None].to(self.device)
        # src_tgt_softmax: how the source aligns to the target
        src_tgt_softmax = torch.softmax(alignment_scores + src_tgt_mask, dim=-1)
        tgt_src_softmax = torch.softmax(alignment_scores + tgt_src_mask, dim=-2)

        token_level_alignment_mat_labels = self.construct_token_level_alignment_mat_from_word_level_alignment_list(**batch.alignment_construction_params)
        # div by len is default for now:
        # note: in their implementation of div by len I think they have swapped the src and tgt lens. note on this note: I think their div by len is actually right nvm.
        # turns out flatten.sum isn't significantly slower on mps or cpu than .sum((1,2))
        per_batch_loss = -(token_level_alignment_mat_labels * src_tgt_softmax).flatten(1).sum(1) / len_src \
             - (token_level_alignment_mat_labels * tgt_src_softmax).flatten(1).sum(1) / len_tgt
        return {"loss_per_batch": per_batch_loss, "src_tgt_softmax": src_tgt_softmax, "tgt_src_softmax": tgt_src_softmax}
    
    def construct_token_level_alignment_mat_from_word_level_alignment_list(self, gold_possible_word_alignments, src_len, tgt_len, bpe2word_map_src, bpe2word_map_tgt, **kwargs):
        # TODO: remove kwargs and clean up the difference between true labels and the self training labels.
        device = 'cpu' # there many small operations, better to do these on the cpu first, and then move them to gpu 10% speed up in old method, but 50% for my code because I have to create tensors.
        batch_size = len(gold_possible_word_alignments)
        token_level_alignment_mat = torch.zeros((batch_size, src_len, tgt_len), device=device)
        # I rewrote the loop here, but it is less readable, and only causes a 10% speedup. 
        for i in range(batch_size):
            gold_possible_word_alignment = gold_possible_word_alignments[i]
            bpe2word_map_src_i = torch.tensor(bpe2word_map_src[i], device=device)
            bpe2word_map_tgt_i = torch.tensor(bpe2word_map_tgt[i], device=device)
            for src_index_word, tgt_index_word in gold_possible_word_alignment:
                token_level_alignment_mat[i, 1 + torch.where(bpe2word_map_src_i == src_index_word)[0][None, :, None], 1 + torch.where(bpe2word_map_tgt_i == tgt_index_word)[0][None, None, :]] = 1

        return token_level_alignment_mat.to(self.device)

    def get_encoder_hidden_state(self, input_ids):
        from transformers.models.bert.modeling_bert import BertForMaskedLM
        assert isinstance(self.maskedlm, BertForMaskedLM), "this method is specific to the BERTForMaskedLM from huggingface"
        
        activation = dict()
        activation_name = f"layer_{self.layer_number}_activation"
        def hook(module, input, output):
            activation[activation_name] = output[0]
        hook_handle = self.maskedlm.bert.encoder.layer[7].register_forward_hook(hook)

        # run model
        PAD_ID = 0
        attention_mask = (input_ids != PAD_ID).to(self.device)
        self.maskedlm.bert(input_ids, attention_mask)

        hook_handle.remove()
        return activation[activation_name]

    # def get_token_level_alignment(self, src_hidden_states, tgt_hidden_states):
    #     alignments = src_hidden_states @ tgt_hidden_states
    #     return alignments
    
    def get_psi_loss(self, batch: CollateFnReturn):
        # parallel sentence identification loss
        raise NotImplementedError()
        
    def get_mlm_loss(self, batch: CollateFnReturn):
        raise NotImplementedError()
        
    def get_tlm_loss(self, batch: CollateFnReturn):
        raise NotImplementedError()
        
    def get_co_loss(self, batch: CollateFnReturn):
        # consistency optimization loss, and this will likely be combined with so_loss as done in awesome-align.
        raise NotImplementedError() 
    
    def get_bpe_prec_recall_aer_coverage(self,):
        # Should I include this? our hope is to do one to one matching and just add word, phrase, and sentence level encodings, so this won't matter...
        # I will delay writing this. 
        raise NotImplementedError() 

    def get_word_prec_recall_aer_coverage_partial_metrics(self, src_tgt_softmax: torch.Tensor, tgt_src_softmax: torch.Tensor, bpe2word_map_src, bpe2word_map_tgt, gold_possible_word_alignments, gold_sure_word_alignments):
        # computes over a batch based on predictions, returns relevant information to be eventually aggragated
        # prec_recall_aer_coverage
        # need to get the prec, recall, and aer metrics. This is 
        # prec: #possible correct guesses/guesses made = (|H \intersection P| / |H|)
        # recall: #sure correct guesses/possible correct = (|H \intersection S| / |S|)

        # get the token level alignment
        batch_size = src_tgt_softmax.size(0)
        token_alignment = (src_tgt_softmax > self.threshold) * (tgt_src_softmax > self.threshold)
        word_level_alignments = self.get_word_alignment_from_token_alignment(token_alignment, bpe2word_map_src, bpe2word_map_tgt)

        num_sure = 0
        guesses_made = 0
        guesses_made_in_possible = 0
        guesses_made_in_sure = 0
        for i in range(batch_size):
            word_level_alignment = word_level_alignments[i]
            gold_possible_word_alignment = gold_possible_word_alignments[i]
            gold_sure_word_alignment = gold_sure_word_alignments[i]

            num_sure += len(gold_sure_word_alignment)
            guesses_made += len(word_level_alignment)
            guesses_made_in_sure += len(set(word_level_alignment).intersection(gold_sure_word_alignment))
            guesses_made_in_possible += len(set(word_level_alignment).intersection(gold_possible_word_alignment))
        prec_recall_aer_coverage_partial_metrics = {
            "num_sure": num_sure,
            "guesses_made": guesses_made,
            "guesses_made_in_sure": guesses_made_in_sure,
            "guesses_made_in_possible": guesses_made_in_possible,
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
        losses = self.training_step(batch)
        return losses
    def test_step(self, batch: CollateFnReturn):
        return self.val_step(batch)
    def predict_step(self, batch: CollateFnReturn):
        pass
    # def configure_optimizers(self): 
    #     # pytorch lightning does this, but I don't get it, so will just do the init of the optimizer and scheduler inside the trainer.
    #     # update: I think I should have this now. the model is related to how it optimizes itself. It knows which parameters of itself 
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

            bpe2word_map_src.append(example.bpe2word_map_src)
            bpe2word_map_tgt.append(example.bpe2word_map_tgt)
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
