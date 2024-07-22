from abc import abstractmethod
import torch.nn as nn
import torch
from typing import List, Dict, Any
from dataclasses import dataclass
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

# class AwesomeAlignerValReturn(UserDict):
#     def update(self, other=None, **kwargs):
#         # TODO: this needs a more reasonable update function where the val parameters will be treated differently depending on what they are...
#         # thus our aer, and prec recall can be handled correctly. Might look at torchmetrics.
#         # def update(self, m,  **kwargs):
#         ...

        
class AwesomeAligner(BaseTextAligner):
    def __init__(self, maskedlm: nn.Module, layer_number: int, device: str):
        super().__init__()
        self.maskedlm = maskedlm.to(device) # for some models you can initialize them on the device. Not bert sadly.
        self.layer_number = layer_number
        self.device = device
        # the per encoder way to get layer hidden rep will be different. 
        # Some could possibly be achieved by the forward hook, others through what awesomealign used (rewriting the encode function to accept a layer parameter)
    def forward(self, x):
        pass


    def training_step(self, batch: CollateFnReturn):
        # a dict is chosen here for our return type because most of the metrics we just want to treat uniformly, and ease of adding values is important. Type safety not so much so.

        losses = dict()
        losses["loss"] = self.get_supervised_training_loss(batch)
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
        token_level_alignment_mat_labels = self.construct_token_level_alignment_mat_from_word_level_alignment_list(**batch.alignment_construction_params)
        src_hidden_states = self.get_encoder_hidden_state(batch.examples_src)
        tgt_hidden_states = self.get_encoder_hidden_state(batch.examples_tgt)
        alignment_scores = (src_hidden_states @ tgt_hidden_states.transpose(-1,-2))
        src_tgt_mask = ((batch.examples_tgt == CLS_ID) | (batch.examples_tgt == SEP_ID) | (batch.examples_tgt == PAD_ID)).to(self.device)
        tgt_src_mask = ((batch.examples_src == CLS_ID) | (batch.examples_src == SEP_ID) | (batch.examples_src == PAD_ID)).to(self.device)
        len_src = src_tgt_mask.sum(1)
        len_tgt = tgt_src_mask.sum(1)

        
        src_tgt_mask = (src_tgt_mask * torch.finfo(torch.float32).min)[:, None, :].to(self.device)
        tgt_src_mask = (tgt_src_mask * torch.finfo(torch.float32).min)[:, :, None].to(self.device)
        # said: how does the source align to the target for src_tgt_softmax
        src_tgt_softmax = torch.softmax(alignment_scores + src_tgt_mask, dim=-1)
        tgt_src_softmax = torch.softmax(alignment_scores + tgt_src_mask, dim=-2)

        # div by len is default for now:
        # note: in their implementation of div by len I think they have swapped the src and tgt lens.
        # turns out flatten.sum isn't significantly slower on mps or cpu than .sum((1,2))
        per_batch_loss = -(token_level_alignment_mat_labels * src_tgt_softmax).flatten(1).sum(1) / len_tgt \
             - (token_level_alignment_mat_labels * tgt_src_softmax).flatten(1).sum(1) / len_src
        return per_batch_loss.mean()
    
    def construct_token_level_alignment_mat_from_word_level_alignment_list(self, word_aligns, src_len, tgt_len, bpe2word_map_src, bpe2word_map_tgt, **kwargs):
        # TODO: remove kwargs and clean up the difference between true labels and the self training labels.
        device = "cpu" # there are going to be many small operations, perhaps better to do these on the cpu first, and then move them to gpu
        batch_size = len(word_aligns)
        token_level_alignment_mat = torch.zeros((batch_size, src_len, tgt_len), device=device)
        for i in range(batch_size):
            word_aligns_i = word_aligns[i]
            bpe2word_map_src_i = torch.tensor(bpe2word_map_src[i], device=device)
            bpe2word_map_tgt_i = torch.tensor(bpe2word_map_tgt[i], device=device)
            for src_index_word, tgt_index_word in word_aligns_i:
                token_level_alignment_mat[i, 1 + torch.where(bpe2word_map_src_i == src_index_word)[0][None, :, None], 1 + torch.where(bpe2word_map_tgt_i == tgt_index_word)[0][None, None, :]] = 1
        # for idx, (word_align, b2w_src, b2w_tgt) in enumerate(zip(word_aligns, bpe2word_map_src, bpe2word_map_tgt)):
        #     len_src = min(bpelen_src, len(b2w_src))
        #     len_tgt = min(bpelen_tgt, len(b2w_tgt))
        #     for i in range(len_src):
        #         for j in range(len_tgt):
        #             if (b2w_src[i], b2w_tgt[j]) in word_align:
        #                 guide[idx, 0, i+1, j+1] = 1.0

        
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
    def collate_fn(examples: List[AwesomeAlignDatasetReturn]):
        examples_src, examples_tgt, examples_srctgt, examples_tgtsrc, langid_srctgt, langid_tgtsrc, psi_examples_srctgt, psi_labels = [], [], [], [], [], [], [], []
        src_len = tgt_len = 0
        bpe2word_map_src, bpe2word_map_tgt = [], []
        word_aligns = []
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
            word_aligns.append(example.possible_labels)
        examples_src = pad_sequence(examples_src, batch_first=True, padding_value=pad_token_id)
        examples_tgt = pad_sequence(examples_tgt, batch_first=True, padding_value=pad_token_id)
        examples_srctgt = pad_sequence(examples_srctgt, batch_first=True, padding_value=pad_token_id)
        langid_srctgt = pad_sequence(langid_srctgt, batch_first=True, padding_value=pad_token_id)
        examples_tgtsrc = pad_sequence(examples_tgtsrc, batch_first=True, padding_value=pad_token_id)
        langid_tgtsrc = pad_sequence(langid_tgtsrc, batch_first=True, padding_value=pad_token_id)
        psi_examples_srctgt = pad_sequence(psi_examples_srctgt, batch_first=True, padding_value=pad_token_id)
        psi_labels = torch.tensor(psi_labels)
        if word_aligns[0] is None:
            word_aligns = None

        # at some point I have to create the alignments between the source and the target through the model itself, and I have the necessary parameters here for doing that, so I should just give what parameters I have to the function which is in charege of that
        alignment_construction_params = dict(inputs_src=examples_src, inputs_tgt=examples_tgt, bpe2word_map_src=bpe2word_map_src, bpe2word_map_tgt=bpe2word_map_tgt, src_len=src_len, tgt_len=tgt_len, word_aligns=word_aligns)
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
    # test the collate function
    # from torch.utils.data import DataLoader
    # from greek.datasetloaders.datasetloaders import AwesomeAlignDataset
    # from transformers import AutoTokenizer
    # from tqdm import tqdm
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    # # src_tgt_file = "/Users/jakob/dev/greek/data/awesome_modified_data/enfr/test_enfr.src-tgt"
    # # gold_labels = "/Users/jakob/dev/greek/data/awesome_modified_data/enfr/test_enfr.gold"
    # src_tgt_file = "/Users/jakob/dev/greek/data/awesome_training_data/multilingual_data_nozh.src-tgt"
    # gold_labels = None
    # pad_token_id = tokenizer.pad_token_id
    # dataset = AwesomeAlignDataset(tokenizer=tokenizer, src_tgt_file=src_tgt_file, gold_file=gold_labels, gold_one_index=True, ignore_possible_alignments=False)
    # test_loader = DataLoader(dataset, num_workers=0, batch_size=4, collate_fn=get_collate_fn(pad_token_id, block_size=512))
    # for i, batch in enumerate(tqdm(test_loader)):
    #     # print(batch)
    #     pass
    pass
    # #%%
    # from transformers import AutoTokenizer
    # maskedlm = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    # maskedlm.model_max_length
    # # %%
