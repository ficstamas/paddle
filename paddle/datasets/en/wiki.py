import requests
import os
from typing import Optional, Literal, Callable
import tqdm
import zipfile
import time
import re
from paddle.utils.network import download_url


__LANGUAGES = [
    "aa", "ab", "ace", "ady", "af", "ak", "als", "am", "an", "ang", "ar", "arc",
    "arz", "as", "ast", "atj", "av", "ay", "az", "azb", "ba", "bar", "bat-smg",
    "bcl", "be", "be-x-old", "bg", "bh", "bi", "bjn", "bm", "bn", "bo", "bpy",
    "br", "bs", "bug", "bxr", "ca", "cbk-zam", "cdo", "ce", "ceb", "ch", "cho",
    "chr", "chy", "ckb", "co", "cr", "crh", "cs", "csb", "cu", "cv", "cy", "da",
    "de", "din", "diq", "dsb", "dty", "dv", "dz", "ee", "el", "eml", "en", "eo",
    "es", "et", "eu", "ext", "fa", "ff", "fi", "fiu-vro", "fj", "fo", "fr",
    "frp", "frr", "fur", "fy", "ga", "gag", "gan", "gd", "gl", "glk", "gn",
    "gom", "gor", "got", "gu", "gv", "ha", "hak", "haw", "he", "hi", "hif",
    "ho", "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ig", "ii",
    "ik", "ilo", "inh", "io", "is", "it", "iu", "ja", "jam", "jbo", "jv", "ka",
    "kaa", "kab", "kbd", "kbp", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko",
    "koi", "krc", "ks", "ksh", "ku", "kv", "kw", "ky", "la", "lad", "lb",
    "lbe", "lez", "lfn", "lg", "li", "lij", "lmo", "ln", "lo", "lrc", "lt",
    "ltg", "lv", "mai", "map-bms", "mdf", "mg", "mh", "mhr", "mi", "min", "mk",
    "ml", "mn", "mr", "mrj", "ms", "mt", "mus", "mwl", "my", "myv", "mzn", "na",
    "nah", "nap", "nds", "nds-nl", "ne", "new", "ng", "nl", "nn", "no", "nov",
    "nrm", "nso", "nv", "ny", "oc", "olo", "om", "or", "os", "pa", "pag", "pam",
    "pap", "pcd", "pdc", "pfl", "pi", "pih", "pl", "pms", "pnb", "pnt", "ps",
    "pt", "qu", "rm", "rmy", "rn", "ro", "roa-rup", "roa-tara", "ru", "rue",
    "rw", "sa", "sah", "sat", "sc", "scn", "sco", "sd", "se", "sg", "sh", "si",
    "simple", "sk", "sl", "sm", "sn", "so", "sq", "sr", "srn", "ss", "st",
    "stq", "su", "sv", "sw", "szl", "ta", "tcy", "te", "tet", "tg", "th", "ti",
    "tk", "tl", "tn", "to", "tpi", "tr", "ts", "tt", "tum", "tw", "ty", "tyv",
    "udm", "ug", "uk", "ur", "uz", "ve", "vec", "vep", "vi", "vls", "vo", "wa",
    "war", "wo", "wuu", "xal", "xh", "xmf", "yi", "yo", "za", "zea", "zh",
    "zh-classical", "zh-min-nan", "zh-yue", "zu"]


def _wiki_url(lang, date):
    return f"https://dumps.wikimedia.org/{lang}wiki/{date}/{lang}wiki-{date}-pages-articles.xml.bz2"


def _acquire_resource(name, path):
    """
    Downloads resource if not exists
    :param name: Name of the resource specified in _URLS dict
    :param path: Path to the ZIP file
    :return:
    """
    if not os.path.isfile(path):
        return download(name, path)
    return path


def _remove_paragraph_format(title: str):
    return title.strip("= ")


def clean_wikitext(name: Literal['wikitext2', 'wikitext103'], path: str):
    """
    Cleans 'wikitext2' and 'wikitext103' data
    :param name: Name of the resource specified in _URLS dict
    :param path: Path to the ZIP file
    :return:
    """
    path = _acquire_resource(name, path)
    if name == 'wikitext2':
        prefix = 'wikitext-2'
    elif name == 'wikitext103':
        prefix = 'wikitext-103'
    else:
        raise NotImplementedError(f"Cleaning function is not implemented for {name}")
    with zipfile.ZipFile(path) as zf:
        train_data = zf.read(f'{prefix}-raw/wiki.train.raw')
        valid_data = zf.read(f'{prefix}-raw/wiki.valid.raw')
        test_data = zf.read(f'{prefix}-raw/wiki.test.raw')

        base_path = os.path.split(path)[0]
        folder = os.path.join(base_path, name)

        regex = re.compile(r"^(= [^=]* =)$")  # detecting new document
        regex_2 = re.compile(r" @.?@ ")  # to find splits of continuous tokens e.g. `10 @.@ 4` -> `10.4`

        for filename, part in [('train.txt', train_data.decode("utf8")),
                               ('valid.txt', valid_data.decode("utf8")),
                               ('test.txt', test_data.decode("utf8"))]:
            file = os.path.join(folder, filename)

            os.makedirs(folder, exist_ok=True)
            f = open(file, mode="w", encoding="utf8")

            lines = part.split('\n')
            print(f"Cleaning {name}:{filename} data...")
            time.sleep(.2)  # :) just a lazy attempt to fix the stdout

            # Cleaning is a simple process
            # We are just removing the control bytes, potential spaces, and
            # reconstructing continuous tokens e.g. `10 @.@ 4` -> `10.4`
            # Appending to a .txt file line by line
            for line in tqdm.tqdm(lines):
                l_: str = line.rstrip('\n\r\t')
                l_ = l_.strip(' ')
                for pattern in re.findall(regex_2, l_):
                    replace_character = pattern[2]
                    l_ = l_.replace(pattern, replace_character)
                if l_.__len__() == 0:
                    continue
                if l_.startswith("=") and l_.endswith("="):
                    if re.match(regex, l_):
                        f.write(f"\n{_remove_paragraph_format(l_)}\n")
                        continue
                    f.write(f"{_remove_paragraph_format(l_)}\n")
                    continue
                f.write(f"{l_}\n")
            f.close()


_URLS = {
    'wikitext2': {
        'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip',
        'clean': clean_wikitext
    },
    'wikitext103': {
        'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip',
        'clean': clean_wikitext
    },
    'wikidump': {
        'url': 'http://mattmahoney.net/dc/enwik8.zip',
        'clean': None
    }
}

_RESOURCE_TYPES = Literal['wikitext2', 'wikitext103', 'wikidump']


def download(name: _RESOURCE_TYPES,
             path: Optional[str],
             retries: Optional[int] = 5,
             verify_ssl: Optional[bool] = True) -> Optional[str]:
    """
    Downloads Resource

    :param name: Name of the resource specified in _URLS dict
    :param path: Destination
    :param retries: Maximum number of retries to acquire the resource
    :param verify_ssl: Verify SSL certificates
    :return: Path to the resource, or None
    """
    assert name in _URLS, f"The provided resource {name} is not available. Consider using the following alternatives: {', '.join(_URLS)}"
    assert retries >= 0, f"Number of retries should be at least 0, currently it's {retries}"
    if not os.path.isdir(path):
        path = os.path.split(path)[0]

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    url = _URLS[name]['url']
    file_name = url.split('/')[-1]
    file = os.path.join(path, file_name)

    file = download_url(url, file, retries, verify_ssl)

    return file


def download_wikidump(path: Optional[str],
                      lang: Optional[str],
                      date: Optional[str],
                      retries: Optional[int] = 5,
                      verify_ssl: Optional[bool] = True) -> Optional[str]:
    """
    Downloads wikipedia dump to location

    :param path: Path on drive
    :param lang: Language Code. See _LANGUAGES
    :param date: Format: YYYYMMDD, or 'latest'
    :param retries: Maximum number of retries to acquire the resource
    :param verify_ssl: Verify SSL certificates
    :return: Path to the resource, or None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if lang not in __LANGUAGES:
        raise ValueError('Unsupported language code')
    language = lang.replace('-', '_')
    output_file = os.path.join(path, 'download', language, date, 'wikicorpus.xml.bz2')

    url = _wiki_url(language, date)
    output_file = download_url(url, output_file, retries, verify_ssl)

    return output_file


def preprocess_resource(name: _RESOURCE_TYPES,
                        path: Optional[str]):
    """
    Cleans and preprocesses data

    :param name: Name of the resource to use the appropriate cleaning function
    :param path: Path to the ZIP
    :return:
    """
    assert name in _URLS, f"The provided resource {name} is not available. Consider using the following alternatives: {', '.join(_URLS)}"
    clean_function: Optional[Callable] = _URLS[name]['clean']
    if clean_function is None:
        raise NotImplementedError(f"Preprocessing function is not implemented for {name}")
    clean_function(name, path)


def load_dataset(name: _RESOURCE_TYPES,
                 path: Optional[str]):
    """
    Loads dataset

    :param name: Name of the resource specified in _URLS dict
    :param path: Destination
    :return:
    """
    if name == 'wikidump':
        raise NotImplementedError()
