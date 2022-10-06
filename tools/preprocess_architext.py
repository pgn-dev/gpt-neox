
"""Processing architext dataset for FIM, and later diffing."""

import argparse
import multiprocessing
import os
import sys

import lm_dataformat as lmd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
import tqdm
import numpy as np
import torch

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from threading import Semaphore


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = Encoder.tokenizer.tokenize(text)
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(text)

class FIMer(object):
    def __init__(self, args):
        self.args = args
    
    def initializer(self, seed_sequence):
        # use FIMer class as a global rng
        FIMer.rng = np.random.default_rng(seed_sequence)

    def fim_document(self, text):
        fim_rate = self.args.fim_rate
        
        suffix_token, prefix_token, middle_token = self.args.suffix_token, self.args.prefix_token, self.args.middle_token
        eod_token = "<|endoftext|>"

        if FIMer.rng.binomial(1, fim_rate):
            # FIM this document
            # Remove <EOD> token
            text = text.replace(eod_token, "")

            # error handling for empty documents/lines
            try:
                # Split text into 2 parts
                prompt, layout = text.split("[layout]")
                spaces = layout.split(", ")
                spaces = [s.strip() for s in spaces]
                if len(spaces) <= 2:
                    # don't bother masking
                    return text, len(text)
                # shuffle the spaces
                FIMer.rng.shuffle(spaces)

                # sample a number of spaces to mask, 
                # mask at least one, keep at least one ?
                num_spaces = FIMer.rng.integers(1, len(spaces)-1)

                # select what contiguous spaces to mask
                # don't mask the first one

                start_idx = FIMer.rng.integers(1, len(spaces)-num_spaces)
                end_idx = start_idx + num_spaces

                prefix = prompt + "[layout] " + prefix_token + " " + ", ".join(spaces[:start_idx])
                middle = middle_token + " " + ", ".join(spaces[start_idx:end_idx])
                suffix = suffix_token + " " + ", ".join(spaces[end_idx:]) + " "

                # PSM or SPM?
                # do PSM for now
                text = " ".join([prefix, suffix, middle, eod_token])
            except Exception as e:
                safe_text = ' '.join('{:02X}'.format(c) for c in text)
                print(f"Error FIMing document {safe_text}: {e}")
                return '', 0
        return text, len(text)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input files that contain prompts and layouts on each line"
        "list",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--prefix-token",
        type=str,
        default="<|prefix|>",
        help="Token to use for prefix (during FIM).",
    )
    group.add_argument(
        "--middle-token",
        type=str,
        default="<|middle|>",
        help="Token to use for middle (during FIM).",
    )
    group.add_argument(
        "--suffix-token",
        type=str,
        default="<|suffix|>",
        help="Token to use for suffix (during FIM).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument(
        "--fim-rate",
        type=float,
        default=1.0,
        help="Rate at which to apply FIM preproc.",
    )

    group = parser.add_argument_group(title="output data")

    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to output file(s) without suffix",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )
    group.add_argument(
        "--output-vocabulary-dir",
        type=str,
        default=None,
        help="Optional path to save the vocabulary used to encode the data",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            for doc in f.strip().split('\n'):
                yield doc

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def main():
    args = get_args()
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
    # hence building up memory
    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents
    fin = yield_from_files(args.input.split(","), semaphore)

    seed_seq = np.random.SeedSequence(1234)

    fimer = FIMer(args)

    # import pdb; pdb.set_trace()
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=fimer.initializer, initargs=(seed_seq,))
        fimed_docs = pool.imap(fimer.fim_document, fin, chunksize=25)
    else:
        fimer.initializer(seed_seq)
        fimed_docs = (fimer.fim_document(doc) for doc in fin)

    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    with open(f"{args.output_prefix}.txt", "w") as fout:
        for i, (doc, bytes_processed) in enumerate(fimed_docs, start=1):
            total_bytes_processed += bytes_processed
            semaphore.release()
            if doc:
                fout.write(doc + "\n")
            # log progress
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                pbar.set_description(
                    f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed} docs/s, {mbs} MB/s)."
                )
                if i != 0:
                    pbar.update(args.log_interval)

    sys.exit()
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    # make a dataset builder for each key in args.jsonl_keys
    # each key will output to a different file beginning with args.output_prefix
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(
            args.output_prefix, key, "document"
        )
        output_idx_files[key] = "{}_{}_{}.idx".format(
            args.output_prefix, key, "document"
        )
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            # separate with eos token
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed} docs/s, {mbs} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
