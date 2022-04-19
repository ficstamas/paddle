from typing import Optional
from paddle.utils.network import download_url
import os
import pandas as pd
from paddle.datasets.dataclasses import DataSplitsOpinHuBank
import zipfile
from collections import Counter
import tqdm


_RESOURCE_URL = 'http://metashare.nytud.hu/repository/download/608756be64e211e2aa7c68b599c26a068dd5b3551f024f6281131670412d37d3/'
_CITATION = 'Miháltz, Márton (2013). “OpinHuBank: szabadon hozzáférhető annotált korpusz magyar nyelvű véleményelemzéshez”. ' \
            'Tanács Attila, Vincze Veronika (szerk.): IX. Magyar Számítógépes Nyelvészeti Konferencia (MSZNY 2013), SZTE, Szeged, 2013, pp. 343-345.'


def _aggregate_labels(df: pd.DataFrame):
    cols = [f'Annot{i}' for i in range(1, 6)]
    labels = pd.DataFrame(index=df.index, columns=['labels'])
    for i in tqdm.tqdm(df.index, desc='Aggregating Labels'):
        annotations = df.loc[i][cols].values
        freq = Counter(annotations).most_common()
        if len(freq) > 1 and freq[0][1] == 2 and freq[1][1] == 2:
            freq = 0
        else:
            freq = freq[0][0]
        labels.loc[i]['labels'] = freq + 1

    aggregated = pd.concat([df, labels], axis=1)
    return aggregated


def download(path: str,
             retries: int = 5,
             verify_ssl: bool = True) -> Optional[str]:
    """
    Downloads Resource

    :param path: Destination
    :param retries: Maximum number of retries to acquire the resource
    :param verify_ssl: Verify SSL certificates
    :return: Path to the resource, or None
    """
    print("Dataset: ", _CITATION)

    if path.endswith(".zip"):
        path, filename = os.path.split(path)
    else:
        filename = "opinhubank.zip"

    file = os.path.join(path, filename)
    if not os.path.exists(file):
        download_url(_RESOURCE_URL, file, retries, verify_ssl, {'desc': filename}, "post", {
            "licence_agree": "on",
            "in_licence_agree_form": "True",
            "licence": "CC-BY"
        })

    return file


def load_dataset(path: str,
                 download_if_necessary: bool = True,
                 data_split: Optional[list] = (0.7, 0.1, 0.2),
                 random_state: Optional[int] = 42,
                 polar_opinions_only: bool = False) -> DataSplitsOpinHuBank:
    """
    Loads dataset. `labels` column contains aggregated annotations based on majority vote. A label also becomes neutral
    if it has 2 positive, 2 negative and 1 neutral vote.

    :param path: Path to the resource folder
    :param download_if_necessary: Downloads the dataset if it can not found in the provided location
    :param data_split: size of the splits [train, dev, test], if None everything is going to be in the train split
    :param random_state: random state to use for the splits
    :param polar_opinions_only: returns a dataset which contains polar opinions only (removes neutral labels)
    :return: Returns the lists of documents which has been split into lines
    """

    if sum(data_split) != 1:
        raise ValueError("data_split must sum to 1")

    if download_if_necessary:
        path = download(path)

    with zipfile.ZipFile(path) as f:
        f: zipfile.ZipFile
        path, filename = os.path.split(path)
        save_folder = os.path.join(path, filename.rstrip(".zip"))
        os.makedirs(save_folder, exist_ok=True)
        f.extractall(save_folder)

    file = [file for file in os.listdir(save_folder) if file.endswith(".csv")][0]

    df = pd.read_csv(os.path.join(save_folder, file), encoding='iso-8859-2', index_col='ID')
    df = _aggregate_labels(df)

    if polar_opinions_only:
        filter_ = df == 1
        df = df.drop(df[filter_].index)
        
    if data_split is None:
        return DataSplitsOpinHuBank(train=df, test=None, dev=None)

    train = df.sample(frac=data_split[0], random_state=random_state)
    dev = df.drop(train.index).sample(frac=1/(1-data_split[0])*data_split[1], random_state=random_state)
    test = df.drop(train.index).drop(dev.index)

    return DataSplitsOpinHuBank(train=train, test=test, dev=dev)
