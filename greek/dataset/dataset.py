#%%
from typing import Optional, List, Tuple, Set, Any
from collections.abc import Callable
from dataclasses import dataclass
from abc import ABC

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import numpy as np
from tqdm import tqdm
# questions about tokenizing and creating a dataset, and how in depth to make the collate function
# - Should I just create the datasets, and not the dataloaders? what is the trainer in charge of?
#   answer: For pranav, the trainer takes in the config and initializes everything. 
#   For lightning, the trainer in their .fit function take in a train loader and an eval loader. 
#   Also handles saving and loading of optimizer state, and stepping of optimizer, as well as 
#   handling device specific things like distributed stuff...
#   for awesome align's data, sometimes you have labels, and sometimes you do not.
#   in their repo, none of their eval/training has labels. This is a restriction 
#   they make on themselves, but I think we should take some fraction of the alignment 
#   set as a dev set, because just using a heuristic loss isn't going to guide us anywhere 
#   when out test set doesn't corrolate well with the eval signal, and there is no way to 
#   check this without an eval set which resembles the test set at least somewhat.
# - Why not tokenize in the collate function? 
#   answer: we could. and we could tokenize in the get item function of the dataset. 
#   There is no need to preprocess all the data at once. (we don't even use it all!)
# - For collate function, should I include the noise on the tokens?
#   answer: the mdlm clearly wouldn't do that. The typical awesome align does... Since I might want to modify the 
#   noise based on the model's precieved alignments, I would want to add noise maybe seperately for some cases. 
#   I will just always add noise seperately.
# - What happens when collate_fn isn't overwritten for a dataloader on tokenized text, where the tensors are different sizes? 
#   answer: it errors. This question came up because I didn't know where the plaid and mdlm repos were batching. Now it is 
#   clear. plaid batches on the fly on by themselves without the help of a collate function. mdlm uses hf's map function, 
#   and may or may not stream (ie iterate?) the opteration
# - When you implement getitem, how does it work with dataloader processes?, would an iterator work well instead?
#   answer: At this point, the dataset, collate_fn, and worker_init_fn are passed to each worker, where they are used to initialize, and fetch data. 
#   This means that dataset access together with its internal IO, transforms (including collate_fn) runs in the worker process.
#   gotten from https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
# - There is an issue with text data, and functions being passed from worker processes to other processes. Apparently they increment the ref count (https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662), 
#   to combat this issue, the recommendation is to convert the data that you index into with numpy array, and if they are more complex than text good luck! Here we want to use maps from token positions to indices
#   answer: I won't deal with this issue as I have plenty of memory ~1 TB of ram, where the dataset only takes about 1 GB, but the probable best solution is https://github.com/pytorch/pytorch/issues/13246#issuecomment-612396143



def get_word_2_word_gold_alignments(gold_label_str: str, gold_one_index: bool, ignore_possible_alignments: bool):
    # return the sure and possible alignments in a dict, and their structure should be that of a matrix, or it could be just as tuples? 
    # I like the idea of matrix. and for now, just use ignore possible as indicator if you should put the ones in the matrix or not.
    # there is this thing where they extend the matching to the word level if any of the word pieces are aligned. This impacts performance for them massively.
    # and represents a hack because they can't get recall up otherwise. if this is fixed, in that we don't need to do this hack, we would probably be able to 
    # fix phrase alignment more broadly. to extend the labeling they get from their.
    # Alignments are provided on the word level, so they need to be projected to the bpe level for labeling on the self-training objective, but we want them on the 
    # word level when making alignments so that the hack can go through. This hack will in effect take place on phrase matchings aswell? 
    # (maybe not because we won't label particular chunks of words to go together, just embed them and the one to one alignment will be our label still)
    # just need to communicate it. isn't important really. I like the idea of just putting out the matrix, but this would require batching, so won't happen. 
    # since both need to be in the form of word to word at some point, I will just output the same as they have in awesome align.
    possible_labels: Set[Tuple[int, int]] = set() # at the end, possible will be a super set of sure by definition.
    sure_labels: Set[Tuple[int, int]]  = set()
    gold_line = gold_label_str.strip().split()
    for src_tgt in gold_line:
        if 'p' in src_tgt:
            if ignore_possible_alignments:
                continue
            split_char = 'p'
            set_to_add_to = possible_labels
        else:
            split_char = '-'
            set_to_add_to = sure_labels
        wsrc, wtgt = src_tgt.split(split_char)
        wsrc, wtgt = (int(wsrc), int(wtgt)) if not gold_one_index else (int(wsrc)-1, int(wtgt)-1)
        set_to_add_to.add( (wsrc, wtgt) )
    possible_labels.update(sure_labels)
    return sorted(list(possible_labels)), sorted(list(sure_labels))


# help with type hints and changes later on, this was chosen over just a dict (or tuple).
@dataclass
class AwesomeAlignDatasetReturn: 
    src: str
    tgt: str
    src_ids: List[int]
    tgt_ids: List[int]
    bpe2word_map_src: List[int]
    bpe2word_map_tgt: List[Tuple[int, int]]
    negative_sentence_src_ids: List[int]
    negative_sentence_tgt_ids: List[int]
    possible_labels: Optional[List[Tuple[int, int]]] = None
    sure_labels: Optional[List[Tuple[int, int]]] = None

def identity_preprocessing(src_tgts: List[Tuple[str,str]], gold_lines: Any) -> List[Tuple[str,str]]:
    return src_tgts
# from typing import List, Tuple, Any
# import numpy as np
# from pprint import pprint
def preprocessing(src_tgt_pairs: List[Tuple[str,str]], gold_lines: Any, prob_combine: float, prob_delete: float, prob_swap: float) -> List[Tuple[str,str]]:
    assert (gold_lines is None) or (prob_combine == prob_delete == prob_swap == 0), "Don't support permuting labels"
    # Swap Combines Deletions
    # prob_combine = 0.5
    # prob_delete = 0.2
    # prob_swap = 0.2
    src_tgt_list_pairs = [([src], [tgt]) for src, tgt in src_tgt_pairs]
    # first combine some up to 3 sentences?
    new_src_tgts = [src_tgt_list_pairs[0]]
    i = 0
    while i < len(src_tgt_list_pairs)-1:
        src_list, tgt_list = new_src_tgts[-1]
        if prob_combine > np.random.rand() and len(src_list) < 3:
            src_list = src_list + src_tgt_list_pairs[i + 1][0]
            tgt_list = tgt_list + src_tgt_list_pairs[i + 1][1]
            new_src_tgts[-1] = (src_list, tgt_list)
        else:
            new_src_tgts.append(src_tgt_list_pairs[i+1])
        i += 1
    # pprint([(" ".join(src), " ".join(tgt)) for src, tgt in new_src_tgts])
    # then delete some of them
    src_tgt_list_pairs = new_src_tgts
    new_src_tgts = []
    i = 0
    while i < len(src_tgt_list_pairs):
        src_list, tgt_list = src_tgt_list_pairs[i]
        if prob_delete > np.random.rand() and len(src_list) > 1:
            index_to_del = np.random.randint(0,len(src_list))
            if 0.5 > np.random.rand(): # delete in src
                src_list.pop(index_to_del)
            else: # del in tgt
                tgt_list.pop(index_to_del)
        new_src_tgts.append((src_list, tgt_list)) # technically remove is inplace, but whatever.
        i += 1
    # print()
    # pprint([(" ".join(src), " ".join(tgt)) for src, tgt in new_src_tgts])
    # print()
    # then swap some
    src_tgt_list_pairs = new_src_tgts
    new_src_tgts = []
    i = 0
    while i < len(src_tgt_list_pairs):
        src_list, tgt_list = src_tgt_list_pairs[i]
        if prob_swap > np.random.rand():
            if 0.5 > np.random.rand() and len(src_list) > 1: # swap in src
                # I'll just permute as its only 3 items max.
                src_list = [src_list[j] for j in np.random.permutation(len(src_list))]
            elif len(tgt_list) > 1: # swap in tgt
                tgt_list = [tgt_list[j] for j in np.random.permutation(len(tgt_list))]
        new_src_tgts.append((src_list, tgt_list)) # technically remove is inplace, but whatever.
        i += 1
    # pprint([(" ".join(src), " ".join(tgt)) for src, tgt in new_src_tgts])
    new_src_tgts = [(" ".join(src), " ".join(tgt)) for src, tgt in new_src_tgts]
    return new_src_tgts
# preprocessing(list(zip([str(i) for i in range(40)],[str(i) for i in range(40)])), None);
class AwesomeAlignDatasetBase(Dataset[AwesomeAlignDatasetReturn], ABC):
    src_tgt_pairs: List[Tuple[str, str]]
    tokenizer: PreTrainedTokenizerBase
    gold_lines: Any
    gold_one_index: bool
    ignore_possible_alignments: bool

    def __len__(self):
        return len(self.src_tgt_pairs)
    
    def __getitem__(self, i):
        # return src ids, tgt ids, bpe2word map src, bpe2word map tgt, and labels (if available)
        # also for the negative labeling sentence task, should return src and tgt for a negative example.
        neg_i = i
        while neg_i == i:
            neg_i = random.randint(0, self.__len__() - 1)
        # do the logic in here for getting all the info for i, and then just the src and tgt for neg_i.
        src, tgt = self.src_tgt_pairs[i]

        src_token_id_by_words = [self.tokenizer.encode(word, add_special_tokens=False) for word in src.split()]
        tgt_token_id_by_words = [self.tokenizer.encode(word, add_special_tokens=False) for word in tgt.split()]
        bpe2word_map_src = []
        for j, word_list in enumerate(src_token_id_by_words):
            bpe2word_map_src += [j for x in word_list]
        bpe2word_map_tgt = []
        for j, word_list in enumerate(tgt_token_id_by_words):
            bpe2word_map_tgt += [j for x in word_list]
        src_ids = self.tokenizer.encode(src)
        tgt_ids = self.tokenizer.encode(tgt)

        neg_src, neg_tgt = self.src_tgt_pairs[neg_i]
        negative_sentence_src_ids = self.tokenizer.encode(neg_src)
        negative_sentence_tgt_ids = self.tokenizer.encode(neg_tgt)
        example = AwesomeAlignDatasetReturn(
            src=src,
            tgt=tgt,
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            bpe2word_map_src=bpe2word_map_src,
            bpe2word_map_tgt=bpe2word_map_tgt,
            negative_sentence_src_ids=negative_sentence_src_ids,
            negative_sentence_tgt_ids=negative_sentence_tgt_ids,
        )
        if self.gold_lines:
            possible_labels, sure_labels = get_word_2_word_gold_alignments(self.gold_lines[i], self.gold_one_index, self.ignore_possible_alignments)
            example.sure_labels = sure_labels
            example.possible_labels = possible_labels
        return example
    
class AwesomeAlignDatasetMultilingualTraining(AwesomeAlignDatasetBase):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase,
                 src_tgt_enfr_file: str,
                 src_tgt_roen_file: str,
                 src_tgt_deen_file: str,
                 src_tgt_jaen_file: str,
                 preprocessing: Callable[[List[Tuple[str,str]],Any], List[Tuple[str,str]]],
                 len_per_lang: int):
        self.tokenizer = tokenizer
        self.gold_lines = None
        self.gold_one_index = False
        self.ignore_possible_alignments = False
        # opens all the files independantly and preprocesses them, and then combines them into one long thing.
        self.src_tgt_pairs = []
        for file_name in tqdm([src_tgt_enfr_file, src_tgt_roen_file, src_tgt_deen_file, src_tgt_jaen_file], desc="MultiLingual Dataprep"):
            src_tgt_pairs_one_lang = []
            with open(file_name, "r", encoding="utf-8") as fin:
                src_tgt_lines = fin.readlines()
                i = 0
                for src_tgt_line in src_tgt_lines:
                    src, tgt = src_tgt_line.split(" ||| ")
                    src = src.strip()
                    tgt = tgt.strip()
                    if len(src) <= 0 or len(tgt) <= 0:
                        continue
                    i += 1
                    src_tgt_pairs_one_lang.append((src, tgt))
                    if i >= len_per_lang:
                        break
            self.src_tgt_pairs += preprocessing(src_tgt_pairs_one_lang, None)

class AwesomeAlignDataset(AwesomeAlignDatasetBase):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase, 
                 src_tgt_file: str, 
                 gold_file: Optional[str], 
                 gold_one_index: bool, 
                 ignore_possible_alignments: bool,
                 preprocessing: Callable[[List[Tuple[str,str]],Any], List[Tuple[str,str]]]):
        self.tokenizer = tokenizer
        src_tgt_lines = open(src_tgt_file, encoding="utf-8").readlines()

        self.gold_lines = None
        if gold_file:
            self.gold_lines = open(gold_file, encoding="utf-8").readlines()
            assert len(self.gold_lines) == len(src_tgt_lines), "must be the same length files if you are to use one as labels for the other."
        # then filter based on src tgt sentence length
        self.src_tgt_pairs = []
        subset_indices = []
        for i, src_tgt_line in enumerate(src_tgt_lines):
            src, tgt = src_tgt_line.split(" ||| ")
            src = src.strip()
            tgt = tgt.strip()
            if len(src) <= 0 or len(tgt) <= 0:
                continue
            self.src_tgt_pairs.append((src, tgt))
            subset_indices.append(i)

        if self.gold_lines:
            self.gold_lines = [self.gold_lines[i] for i in subset_indices]
        self.src_tgt_pairs = preprocessing(self.src_tgt_pairs, self.gold_lines)
        self.gold_one_index = gold_one_index
        self.ignore_possible_alignments = ignore_possible_alignments


class AwesomeAlignDatasetsMap:
    def __init__(self, enfr_dataset, deen_dataset, roen_dataset, jaen_dataset):
        self.enfr_dataset = enfr_dataset
        self.deen_dataset = deen_dataset
        self.roen_dataset = roen_dataset
        self.jaen_dataset = jaen_dataset
        self.map = dict()
        if enfr_dataset is not None:
            self.map["enfr"] = enfr_dataset
        if roen_dataset is not None:
            self.map["roen"] = roen_dataset
        if deen_dataset is not None:
            self.map["deen"] = deen_dataset
        if jaen_dataset is not None:
            self.map["jaen"] = jaen_dataset
    def get(self, dataset_name):
        return self.map[dataset_name]
    def keys(self):
        return self.map.keys()
    

    
    
# the dataset handles the data, and therefor is highly related to several 
# implementation details which stem from where you would tokenize the data or perform other data processing opterations.
# our model may choose to incorperate a new loss like the psi loss in the paper,
# and our dataloader's collate function must adjust, but also our dataset must
# add the ability to get negatives.
# this is different than passing a new param in, as all data is carried at once, like the args object
# so not too much book keeping needs to be done!

# %%
from tqdm import tqdm
import itertools
import transformers
import os
import random
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

# this is the old dataset, but it takes forever to load, which is why they cache it I guess, but this isn't default behavior, and they didn't have it enabled in their readme scripts for whatever reason.
class OLDLineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, cache_data, overwrite_cache, ignore_possible_alignments, gold_one_index, file_path, gold_path, debug):
        assert os.path.isfile(file_path)
        self.debug = debug
        logger.info("Creating features from dataset file at %s", file_path)

        cache_fn = f'{file_path}.cache' if gold_path is None else f'{file_path}.gold.cache'
        if cache_data and os.path.isfile(cache_fn) and not overwrite_cache:
            logger.info("Loading cached data from %s", cache_fn)
            self.examples = torch.load(cache_fn)
        else:
            # Loading text data
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                if self.debug:
                    print(f"\tDEBUG MODE: USING 1000 LINES OF DATA INSTEAD OF {len(lines)} lines from {file_path}.")
                    lines = lines[:1000] # just to make it go faster

            # Loading gold data
            if gold_path is not None:
                assert os.path.isfile(gold_path)
                logger.info("Loading gold alignments at %s", gold_path)
                with open(gold_path, encoding="utf-8") as f:
                    gold_lines = f.readlines()
                assert len(gold_lines) == len(lines)

            for line_id, line in tqdm(enumerate(lines), desc='Loading data', total=len(lines)):
                if len(line) > 0 and not line.isspace() and len(line.split(' ||| ')) == 2:
                    try:
                        src, tgt = line.split(' ||| ')
                        if src.rstrip() == '' or tgt.rstrip() == '':
                            logger.info("Skipping instance %s", line)
                            continue
                    except:
                        logger.info("Skipping instance %s", line)
                        continue
                    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
                    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
                    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

                    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt')['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt')['input_ids'] # type: ignore
                    # print(ids_src)
                    # print(ids_tgt)
                    ids_src = ids_src[None,:] # type: ignore
                    ids_tgt = ids_tgt[None,:] # type: ignore
                    if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
                        logger.info("Skipping instance %s", line)
                        continue

                    bpe2word_map_src = []
                    for i, word_list in enumerate(token_src):
                        bpe2word_map_src += [i for x in word_list]
                    bpe2word_map_tgt = []
                    for i, word_list in enumerate(token_tgt):
                        bpe2word_map_tgt += [i for x in word_list]

                    if gold_path is not None:
                        try:
                            gold_line = gold_lines[line_id].strip().split() # type: ignore
                            gold_word_pairs = []
                            for src_tgt in gold_line:
                                if 'p' in src_tgt:
                                    if ignore_possible_alignments:
                                        continue
                                    wsrc, wtgt = src_tgt.split('p')
                                else:
                                    wsrc, wtgt = src_tgt.split('-')
                                wsrc, wtgt = (int(wsrc), int(wtgt)) if not gold_one_index else (int(wsrc)-1, int(wtgt)-1)
                                gold_word_pairs.append( (wsrc, wtgt) )
                            self.examples.append( (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, gold_word_pairs) )
                        except:
                            logger.info("Error when processing the gold alignment %s, skipping", gold_lines[line_id].strip()) # type: ignore
                            continue
                    else:
                        self.examples.append( (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, None) )

            if cache_data:
                logger.info("Saving cached data to %s", cache_fn)
                torch.save(self.examples, cache_fn)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        neg_i = random.randint(0, len(self.examples)-1)
        while neg_i == i:
            neg_i = random.randint(0, len(self.examples)-1)
            # if I end up testing with this dataset, I need to reimplement the return type, and change from tensor id types to list of int id types.
        return tuple(list(self.examples[i]) + list(self.examples[neg_i][:2] ) )


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

