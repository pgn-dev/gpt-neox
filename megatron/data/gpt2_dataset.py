# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT2 style dataset."""

import os
import time

import numpy as np
import torch

from megatron import mpu, print_rank_0


class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=True,
        neox_args=None,
    ):
        # TODO (hailey:) add a check that noise_density and mean_noise_span_length are not passed to GPT2Dataset init?
        self.name = name
        self.seed = seed
        self.indexed_dataset = indexed_dataset
        self.neox_args = neox_args

        self.np_rng = np.random.RandomState(seed=seed) # rng state for FIM

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        if build_index_mappings:
            # Build index mappings.
            self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
                self.name,
                data_prefix,
                documents,
                self.indexed_dataset.sizes,
                num_samples,
                seq_length,
                seed,
            )
            self.shuffle_idx_len = self.shuffle_idx.shape[0] - 1
            self.sample_idx_len = self.sample_idx.shape[0] - 1

            if self.shuffle_idx_len != self.sample_idx_len:
                print(
                    f"WARNING: shuffle index length ({self.shuffle_idx_len}) is not equal to sample index length ({self.sample_idx_len})"
                )

    def __len__(self):
        return min(self.shuffle_idx_len, self.sample_idx_len)

    def __getitem__(self, idx):
        try:
            # Get the shuffled index.
            idx = self.shuffle_idx[idx]
            # Start and end documents and offsets.
            doc_index_f = self.sample_idx[idx][0]
            doc_index_l = self.sample_idx[idx + 1][0]
            offset_f = self.sample_idx[idx][1]
            offset_l = self.sample_idx[idx + 1][1]
            # If we are within the same document, just extract the chunk.
            if doc_index_f == doc_index_l:
                sample = self.indexed_dataset.get(
                    self.doc_idx[doc_index_f],
                    offset=offset_f,
                    length=offset_l - offset_f + 1,
                )
            else:
                # Otherwise, get the rest of the initial document.
                sample_list = [
                    self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)
                ]
                # Loop over all in between documents and add the entire document.
                for i in range(doc_index_f + 1, doc_index_l):
                    sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
                # And finally add the relevant portion of last document.
                sample_list.append(
                    self.indexed_dataset.get(
                        self.doc_idx[doc_index_l], length=offset_l + 1
                    )
                )
                sample = np.concatenate(sample_list)

            # TODO(Hailey): can merge the code below this line with code above this line.
            # TODO(Hailey), cont: above already iterates through loop, so just add the permuting in there?
            sample = np.array(sample, dtype=np.int64)
            # # print(sample, sample.shape)
            # # do FIM here, if enabled
            fim_rate = self.neox_args.fim_rate

            if fim_rate != 0:
                assert (fim_rate <= 1 and fim_rate >= 0), "FIM rate must be a probability 0 <= rate <= 1"

                eod = self.neox_args.tokenizer.eod

                segment_breaks = np.argwhere(sample == eod) # split sample by document

                if segment_breaks.shape != (0, 1): # then there is an EOD token in this example
                    curr_start_position = 0
                    for loc in np.nditer(segment_breaks):
                        # print(loc - curr_start_position, flush=True)
                        # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                        # try:
                        if loc - curr_start_position > 10: # sometimes examples start with EOD or are too short. so avoid this case
                            sample[curr_start_position:loc], self.np_rng = \
                                permute(sample[curr_start_position:loc], self.np_rng, self.neox_args)
                        # except ValueError:
                        #     # print(loc - curr_start_position, flush=True)
                        #     pass

                        curr_start_position = loc + 1 # jump over the EOD token
                else:
                    sample, self.np_rng = permute(sample, self.np_rng, self.neox_args)
            
            # end FIM-specific code
            # print(sample, sample.shape)
            return {"text": sample}
            # return {"text": np.array(sample, dtype=np.int64)}

        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


def _build_index_mappings(
    name, data_prefix, documents, sizes, num_samples, seq_length, seed
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):
            print_rank_0(
                " > WARNING: could not find index map files, building "
                "the indices on rank 0 ..."
            )
            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(
                " > elasped time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            from megatron.data import helpers

            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
            )
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                               num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retrieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_io_parallel_group())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=mpu.get_io_parallel_group()
    )

    # Load mappings.
    start_time = time.time()
    print_rank_0(" > loading doc-idx mapping from {}".format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading sample-idx mapping from {}".format(sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading shuffle-idx mapping from {}".format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time)
    )
    print_rank_0("    total number of samples: {}".format(sample_idx.shape[0]))
    print_rank_0("    total number of epochs: {}".format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence length, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng):
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document."""
    doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Beginning offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - 1
                remaining_seq_length = 0
            else:
                # Otherwise, start from the beginning of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(size, np_rng):
    """Build the range [0, size) and shuffle."""
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx

def permute_spaces(sample, np_rng, neox_args):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation
    on spaces in the layout *only* 

    example (ignore newlines): 
    --------------------------------
    [prompt] a bedroom is adjacent to the kitchen 
    [layout] bedroom1: (194,91)(135,91)(135,47)(194,47), 
    living_room: (121,194)(47,194)(47,91)(106,91)(106,106)(121,106), 
    bathroom1: (179,121)(135,121)(135,91)(179,91), 
    bedroom2: (209,165)(135,165)(135,121)(209,121), 
    bathroom2: (165,209)(135,209)(135,165)(165,165), 
    bedroom3: (121,238)(47,238)(47,194)(121,194), 
    kitchen: (135,77)(106,77)(106,18)(135,18), 
    corridor: (121,209)(121,106)(106,106)(106,77)(135,77)(135,209) <|endoftext|>
    --------------------------------

    The transform will mask one or more spaces from the layout (instead of a random span) and place them
    at the end of the prompt. The model can then be tasked with predicting the masked spaces.

    --------------------------------
    <prefix token>
    [prompt] a bedroom is adjacent to the kitchen 
    [layout] bedroom1: (194,91)(135,91)(135,47)(194,47), 
    living_room: (121,194)(47,194)(47,91)(106,91)(106,106)(121,106), 
    <suffix token>
    bedroom2: (209,165)(135,165)(135,121)(209,121), 
    bathroom2: (165,209)(135,209)(135,165)(165,165), 
    bedroom3: (121,238)(47,238)(47,194)(121,194), 
    kitchen: (135,77)(106,77)(106,18)(135,18), 
    corridor: (121,209)(121,106)(106,106)(106,77)(135,77)(135,209) 
    <mask token>
    bathroom1: (179,121)(135,121)(135,91)(179,91),
    <|endoftext|>
    --------------------------------


    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """
    fim_rate = neox_args.fim_rate
    tokenizer = neox_args.tokenizer

    suffix_tok_id, prefix_tok_id, middle_tok_id = tokenizer.vocab_size - 1, tokenizer.vocab_size, tokenizer.vocab_size + 1

    if np_rng.binomial(1, fim_rate): # sample to see if we should transform this sample

        contents = tokenizer.detokenize(sample)
 
        # parse spaces from the layout
        prompt, layout = contents.split("[layout]")
        spaces = layout.split(", ")
        spaces = [s.strip() for s in spaces]
        try:
            # sample a number of spaces to mask, 
            # mask at least one, keep at least one ?
            num_spaces = np_rng.randint(1, len(spaces)-1)

            # select what contiguous spaces to mask
            start_idx = np_rng.randint(0, len(spaces)-num_spaces)
            end_idx = start_idx + num_spaces
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        prefix = prompt + "[layout] " + ", ".join(spaces[:start_idx])
        middle = ", ".join(spaces[start_idx:end_idx])
        suffix = ", ".join(spaces[end_idx:]) + " "

        suffix = np.array([suffix_tok_id, *tokenizer.tokenize(suffix)])
        prefix = np.array([prefix_tok_id, *tokenizer.tokenize(prefix)])
        middle = np.array([middle_tok_id, *tokenizer.tokenize(middle)])

        new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0]
        diff = new_length - sample.shape[0]

        if diff > 0: # too long
            if suffix.shape[0] <= diff: # return the original sample if no space to remove suffix tokens
                return sample, np_rng
            suffix = suffix[:suffix.shape[0] - diff] # truncate suffix tokens (is this desirable?)
        elif diff < 0: # too short
            suffix = np.concatenate([suffix, np.full((-1 * diff), tokenizer.pad)])

        new_sample = np.concatenate([
            prefix,
            suffix,
            middle,
        ])
    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample, np_rng


def permute(sample, np_rng, neox_args):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it. 
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """
    fim_rate = neox_args.fim_rate
    tokenizer = neox_args.tokenizer

    # hardcode these for now. TODO(Hailey): should add a way to access all mask tokens in a tokenizer easily.
    # TODO(Hailey): also, check to ensure there's not an off-by-one error here. 
    # TODO(Hailey): when testing models trained with this workaround, need to add special tokens w/ the correct indices to the tokenizer.
    suffix_tok_id, prefix_tok_id, middle_tok_id = tokenizer.vocab_size - 1, tokenizer.vocab_size, tokenizer.vocab_size + 1

    if np_rng.binomial(1, fim_rate): # sample bernoulli dist

        contents = tokenizer.detokenize(sample)
        
        try:
            boundaries = np_rng.randint(low=1, high=len(contents) - 1, size=2)
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        # print(len(contents), boundaries)

        prefix = contents[:boundaries[0]]
        middle = contents[boundaries[0]:boundaries[1]]
        suffix = contents[boundaries[1]:]

        suffix = np.array([suffix_tok_id, *tokenizer.tokenize(suffix)])
        prefix = np.array([prefix_tok_id, *tokenizer.tokenize(prefix)])
        middle = np.array([middle_tok_id, *tokenizer.tokenize(middle)])
        
        # need to make same length as the input
        new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0]
        diff = new_length - sample.shape[0]

        # print(new_length, sample.shape, suffix.shape, diff)
        if diff > 0: # too long
            if suffix.shape[0] <= diff: # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                return sample, np_rng
            suffix = suffix[:suffix.shape[0] - diff]
        elif diff < 0: # too short
            suffix = np.concatenate([suffix, np.full((-1 * diff), tokenizer.pad)])

        new_sample = np.concatenate([ # TODO(Hailey): add a branch here + a param to select SPM or PSM mode
            suffix,
            prefix,
            middle,
        ])
        # print("definitely using FIM this run")
        # print(new_sample.shape)
        # if sample.shape[0] == 2049 and new_sample.shape[0] != 2049:
        #     diff = new_sample.shape[0] - sample.shape[0]
        #     raise ValueError(f"{boundaries}, {suffix.shape}, {diff}, {new_sample.shape[0]}, {sample.shape[0]}")
        # print(new_sample.shape[0], sample.shape, suffix.shape, diff)
    else:
        # don't do FIM preproc
        new_sample = sample


    return new_sample, np_rng

