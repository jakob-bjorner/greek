#%%
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from greek.dataset.dataset import AwesomeAlignDataset
from greek.model.model import get_collate_fn

# ''' constructs the eval, train, and test split of a particular dataset. Also gives the relevant tokenizer, and '''
# There are some datasources, where I get them from text files, and some where I could get them from online like the parallel data from ted talks.
# These types of data are not the only types of data, needed tho. 
# I should also get potentially noisey sentence alignemnts. 
# This through book level annotation, so the loader will just be the entire book.
# Haven't thought of this well enough, so will likely change a lot during implementation. 
# Just need to leave room for flexibility. so not too type safe. 
class CustTokenizer:
    def __init__(self,pretrained_model_name_or_path):
        print(pretrained_model_name_or_path)

class AwesomeAlignDatasetLoaders:
    '''
    would need to know the collate function to be able to define a dataloader, 
    and this is something that is specific to the data that the model expects 
    to take in.
    '''
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase, 
                 train_dataset: AwesomeAlignDataset, 
                 val_dataset: AwesomeAlignDataset, 
                 test_dataset: AwesomeAlignDataset, 
                 batch_size: int, 
                 num_workers: int, 
                 pin_memory: bool, 
                 pin_memory_device: str):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # AwesomeAlignDataset(tokenizer=self.tokenizer, src_tgt_file="/Users/jakob/dev/greek/data/awesome_training_data/multilingual_data_nozh.src-tgt", gold_file=None, gold_one_index=True, ignore_possible_alignments=False)
        self.tokenizer = tokenizer # AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device

    def setup(self, block_size):
        self.collate_fn = get_collate_fn(pad_token_id=self.tokenizer.pad_token_id, block_size=block_size)
        # self.train_loader = # this can also have labels, or not!
        # self.eval_loader =  # this can have labels, or not...
        # self.test_loader = # this must have lables, and you are expected to get out the AER metric. Could also record the loss from the unsupervised objective losses.

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, pin_memory_device=self.pin_memory_device)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, pin_memory_device=self.pin_memory_device)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, pin_memory_device=self.pin_memory_device) 



if __name__ == "__main__":
    pass
    # cursory testing to ensure correctness and no significant speed decrease in dataset switch from Awesome align to my own.
    # main reason for switch is awesome align preprocesses all their data at once, and I didn't want to do that. 
    # (could have also probably created an iterator for loading the data which could handle the try except gracefully.)

    # from transformers import AutoTokenizer
    # tok = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    # src_tgt_file = "/Users/jakob/dev/greek/data/awesome_modified_data/enfr/test_enfr.src-tgt"
    # gold_labels = "/Users/jakob/dev/greek/data/awesome_modified_data/enfr/test_enfr.gold"
    # src_tgt_file = "/Users/jakob/dev/greek/data/awesome_modified_data/enfr/train_enfr.src-tgt"
    # src_tgt_file = "/Users/jakob/dev/greek/data/awesome_training_data/multilingual_data_nozh.src-tgt"
    # gold_labels = None

    # datasetog = LineByLineTextDataset(tokenizer=tok, overwrite_cache=False, ignore_possible_alignments=False, gold_one_index=True, cache_data=False, file_path=src_tgt_file, gold_path=gold_labels, debug=False)
    # datasetredo = RedoAwesomeAlignDataset(tok, src_tgt_file, gold_labels, True, False)
    # for i in tqdm(range(len(datasetredo))):
    #     datasetredo.__getitem__(i)
    # datasetog.__getitem__(20)

