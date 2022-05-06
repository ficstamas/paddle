from typing import Optional, List, Union, Callable
from paddle.utils.network import get_url_paths, download_url
import re
import os
import tqdm
from paddle.datasets.dataclasses import DataSplits
from datasets import IterableDataset
from datasets.iterable_dataset import ExamplesIterable, _BaseExamplesIterable
from paddle.datasets.utils.datasets import ChainDataset
import numpy as np
from paddle.utils.files import is_gz_file
from gzip import GzipFile
from torch.utils.data import IterableDataset as TorchIterableDataset
import torch


_RESOURCE_URL = 'https://nessie.ilab.sztaki.hu/~ndavid/Webcorpus2_text/'
_CITATION = 'Nemeskey, Dávid Márk (2020). “Natural Language Processing methods for Language Modeling”. ' \
            'PhD thesis. Eötvös Loránd University.'


def download(path: str,
             retries: int = 5,
             verify_ssl: bool = True,
             regex: Optional[str] = None) -> Optional[List]:
    """
    Downloads Resource
    :param path: Destination
    :param retries: Maximum number of retries to acquire the resource
    :param verify_ssl: Verify SSL certificates
    :param regex: Downloads files which match the provided regex e.g. 'wiki*'
    :return: Path to the resource, or None
    """
    print("Dataset: ", _CITATION)
    paths = get_url_paths(_RESOURCE_URL, ext='txt.gz')
    if regex is not None:
        regex = re.compile(regex)
        paths = [p for p in paths if re.match(regex, p.split('/')[-1]) is not None]

    output = []

    for p in paths:
        filename = p.split('/')[-1]
        file = os.path.join(path, filename)
        output.append(file)
        if not os.path.exists(file):
            download_url(p, file, retries, verify_ssl, {'desc': filename})

    return output


def load_dataset(path: str,
                 download_if_necessary: bool = True,
                 regex: Optional[str] = None) -> DataSplits:
    """
    Loads dataset
    :param path: Path to the resource folder
    :param download_if_necessary: Downloads the dataset if it can not found in the provided location
    :param regex: Downloads files which match the provided regex e.g. 'wiki*'
    :return: Returns the lists of documents which has been split into lines
    """

    if download_if_necessary:
        paths = download(path, regex=regex)
    else:
        paths = os.listdir(path)

    output = []
    for p in tqdm.tqdm(paths, desc='Loading'):
        with open(p, mode='r', encoding='utf8') as f:
            lines = f.readlines()
            output.append([l.strip('\n') for l in lines if len(l.strip('\n')) > 0])

    return DataSplits(train=output, test=None, dev=None)


def _generate_lines(file: str):
    """
    Reads file line by line while skipping empty rows
    :param file:
    :returns:
        Yields line
    """
    doc_id = 0
    _, file_ = os.path.split(file)
    if is_gz_file(file):
        with GzipFile(file, mode='r') as f:
            for line in f.readlines():
                stripped = line.decode('utf8').strip('\n')
                if len(stripped) > 0:
                    yield 'sentences', {'text': stripped, 'file': file_, 'doc_id': doc_id}
                else:
                    doc_id += 1
    else:
        with open(file, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                stripped = line.strip('\n')
                if len(stripped) > 0:
                    yield 'sentences', {'text': stripped, 'file': file_, 'doc_id': doc_id}
                else:
                    doc_id += 1


class WebcorpusIterableDataset(TorchIterableDataset):
    def __init__(self, ex_iterators: List[_BaseExamplesIterable], infinite: bool, shuffle_every_cycle: bool,
                 seed: int = 42):
        """

        :param ex_iterators:
        :param infinite: whether to cycle all datasets infinitely
        :param shuffle_every_cycle: whether to shuffle the order of documents every cycle (if `infinite` is True)
        :param seed: Random seed to shuffle documents
        """
        self.ex_iterators = ex_iterators
        self.generator = np.random.default_rng(seed=seed)
        self.infinite = infinite
        self.shuffle_every_cycle = shuffle_every_cycle
        self.dataset: Optional[IterableDataset] = None
        self.map_params = {}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.generator.shuffle(self.ex_iterators)
        if worker_info is None:
            documents = self.ex_iterators
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            documents = [doc for i, doc in enumerate(self.ex_iterators) if i % num_workers == worker_id]
        chain = ChainDataset(documents, self.infinite, self.shuffle_every_cycle, self.generator)
        self.dataset = IterableDataset(chain)
        self.dataset = self.dataset.map(**self.map_params)
        return iter(self.dataset)

    def map(self,
            function: Callable, batched: bool = False, batch_size: int = 1000
            ):
        self.map_params = {
            "function": function,
            "batched": batched,
            "batch_size": batch_size
        }
        return self


def load_iterable_dataset(path: str,
                          download_if_necessary: bool = True,
                          regex: Optional[str] = None,
                          infinite: bool = False,
                          shuffle_every_cycle: bool = False,
                          seed: int = 0) -> WebcorpusIterableDataset:
    """
    Loads dataset
    :param path: Path to the resource folder
    :param download_if_necessary: Downloads the dataset if it can not found in the provided location
    :param regex: Downloads files which match the provided regex e.g. 'wiki*'
    :param infinite: whether to cycle all datasets infinitely
    :param shuffle_every_cycle: whether to shuffle the order of documents every cycle (if `infinite` is True)
    :param seed: Random seed to shuffle documents
    :return: Returns an iterator which cycles the dataset
    """
    if download_if_necessary:
        paths = download(path, regex=regex)
    else:
        paths = os.listdir(path)
        paths = [os.path.join(path, p)for p in paths]

    files = [ExamplesIterable(_generate_lines, {'file': f}) for f in paths]
    iterator = WebcorpusIterableDataset(files, infinite, shuffle_every_cycle, seed)
    return iterator
