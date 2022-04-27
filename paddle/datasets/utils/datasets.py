from datasets.iterable_dataset import _BaseExamplesIterable
import numpy as np
from typing import List
from copy import deepcopy


class ChainDataset(_BaseExamplesIterable):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.
    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.
    """

    @property
    def n_shards(self) -> int:
        return sum(ex_iterable.n_shards for ex_iterable in self.ex_iterables)

    def shuffle_data_sources(self, generator: np.random.Generator) -> "ChainDataset":
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return ChainDataset(ex_iterables, self.infinite, self.shuffle_every_cycle, self.generator)

    def __init__(self, ex_iterables: List[_BaseExamplesIterable], infinite: bool, shuffle_every_cycle: bool,
                 generator: np.random.Generator) -> None:
        """
        :param ex_iterables: (iterable of Iterators): datasets to be chained together
        :param infinite: whether to cycle all datasets infinitely
        :param shuffle_every_cycle: whether to shuffle the order of documents every cycle (if `infinite` is True)
        :param generator: random number generator
        """
        super(ChainDataset, self).__init__()
        self.ex_iterables = ex_iterables
        self.infinite = infinite
        self.shuffle_every_cycle = shuffle_every_cycle
        self.generator = deepcopy(generator)

    def __iter__(self):
        while True:  # loop to cycle
            indices_iterator = range(len(self.ex_iterables))
            iterators = [iter(ex_iterable) for ex_iterable in self.ex_iterables]
            for i in indices_iterator:  # loop to iterate through all files
                while True:  # Infinite loop to iterate file to the end
                    try:
                        yield next(iterators[i])
                    except StopIteration:
                        break
            if not self.infinite:
                break
            if self.shuffle_every_cycle:
                self.ex_iterables = self.shuffle_data_sources(generator=self.generator).ex_iterables

    def __len__(self):
        return len(self.ex_iterables)

