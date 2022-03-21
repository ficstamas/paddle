from typing import Optional, Union, List
import pandas as pd


_COLUMNS_META = "# global.columns = "


def parse_conllu_plus(file: Optional[str] = None, text: Optional[Union[str, List[str]]] = None, encoding: str = "utf8")\
        -> pd.DataFrame:
    """
    Parses Conll-U Plus format into a MultiIndex Pandas Dataframe
    :param file: Path to the conll file
    :param text: Conll file's content
    :param encoding: Encoding of the text
    :return:
        MultiIndex Pandas Dataframe where the first index represents the sentence, and the second the token position
    """
    if file is None and text is None:
        raise ValueError("file or text must have a value other than None!")

    if text is None and file is not None:
        with open(file, mode="r", encoding=encoding) as f:
            text = f.readlines()

    if type(text) is str:
        text = text.split("\n")

    sentence_idx = 0
    token_position = 0

    sent_ = []
    idx_ = []
    cols = None

    for row in text:
        row = row.rstrip("\n")
        # the row contains the meta information about the column names
        if row.startswith(_COLUMNS_META):
            cols = row[len(_COLUMNS_META):].split(' ')
            continue
        # we are going to skip every other meta information
        elif row.startswith("# "):
            continue
        # processing non meta rows
        else:
            # if the row is a separator between sentences
            if len(row) == 0:
                token_position = 0
                sentence_idx += 1
                continue
            values = row.split("\t")
            sent_.append(values)
            idx_.append((sentence_idx, token_position))
            token_position += 1

    # if no column names were found
    if cols is None:
        cols = [f"Col_{i}" for i in range(len(sent_[0]))]

    index = pd.MultiIndex.from_tuples(idx_, names=["sentence_idx_", "token_idx_"])
    df = pd.DataFrame(data=sent_, index=index, columns=cols)
    return df
