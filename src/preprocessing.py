import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.settings import *

DATA_DIR = BASE_DIR / "data"
# 1. Carica modello e tokenizer
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print('Loading model...')
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def slicing_logic(row) -> bool:
    """
    slicing logic for pandas dataframe, leaves only entries with less than 1500 word.
    needed to reduce the dimentions of the dataset and reduce the number of tokens required.
    :param row:
    :return: bool
    """
    if not type(row['article']) is str: return False

    length = len(row['article'].split(' '))
    if  100 < length < 1000:
        return True
    return False


def preprocess_token(batch):
    model_inputs = tokenizer(
        batch["article"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        text_target=batch["abstract"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def concat_dataset():
    """Dataset was already split and kaggle load was not working"""

    df_1 = pd.read_csv(DATA_DIR / 'train.csv')
    df_2 = pd.read_csv(DATA_DIR / 'validation.csv')
    df_3 = pd.read_csv(DATA_DIR / 'test.csv')
    _df = pd.concat([df_1, df_2, df_3])
    _df.to_csv(DATA_DIR / 'dataset.csv', index=False)

def load_csv(path, chunk_size:int) -> pd.DataFrame:
    total_lines = 134 # known form dataset description
    chunks = []
    with pd.read_csv(path, header=0, dtype=str, chunksize=chunk_size) as reader:
        for chunk in tqdm(reader, total=total_lines, desc='Loading Data'):
            chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


# ====== FUNZIONE: Pulizia testo ======
def clean_scientific_text(text):
    if not isinstance(text, str):
        return ""
    # Rimuove spazi multipli
    text = re.sub(r'\s+', ' ', text)
    # Rimuove riferimenti tipo [1], [12], ecc.
    text = re.sub(r'\[\d+\]', '', text)
    # Rimuove caratteri non utili (tranne punteggiatura base)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"()\s-]', '', text)
    # Elimina spazi iniziali/finali
    return text.strip()


class Preprocessor:
    def __init__(self):

        # loading tokenized dataset from cache if available
        if CACHED:
            print("Loading cached dataset...")
            self.tokenized_paths = {
                'train' : DATA_DIR / 'train_tokenized',
                'validation' : DATA_DIR / 'validation_tokenized',
                'test' : DATA_DIR / 'test_tokenized',
            }

            if all(path.exists() for path in self.tokenized_paths.values()):
                self.splits = {
                    name: Dataset.load_from_disk(path)
                    for name, path in self.tokenized_paths.items()
                }

                print(self.splits)
                return
            else:
                print("Cached data was not found. Creating new one...")

        # 2. Carica dataset concatenato
        print("\rLoading dataset...", end='')
        if DEBUG:
            df = pd.read_csv(DATA_DIR / 'dataset.csv', header=0, dtype=str, nrows=2000)
        else:
            df = load_csv(DATA_DIR / 'dataset.csv', 1000)


        print("Reducing dataset...")
        df_filtered = df[df.apply(slicing_logic, axis=1)]
        df_filtered.reset_index(drop=True, inplace=True)

        # 3. Pulizia testo
        for chunk in tqdm(range(len(df_filtered)), desc='Cleaning Dataset'):
            df_filtered.at[chunk, 'article'] = clean_scientific_text(df_filtered.at[chunk, 'article'])
            df_filtered.at[chunk, 'abstract'] = clean_scientific_text(df_filtered.at[chunk, 'abstract'])

        df_clean = df_filtered[df_filtered.apply(slicing_logic, axis=1)]
        df_clean.reset_index(drop=True, inplace=True)
        print(df_clean.shape)


        # 4. Conversione in Dataset HuggingFace
        self.dataset = Dataset.from_pandas(df_filtered)
        print(self.dataset)

        # 5. Split train (80%) / temp (20%)
        print('\rSplitting dataset...', end='')
        self.split_dataset = self.dataset.train_test_split(test_size=0.2, seed=SEED)

        # 6. Split temp in validation (10%) / test (10%)
        self.temp_split = self.split_dataset['test'].train_test_split(test_size=0.5, seed=SEED)

        self.dataset_splits = {
            'train': self.split_dataset['train'],
            'validation': self.temp_split['train'],
            'test': self.temp_split['test']
        }

        print('split done')
        # 8. Tokenizzazione di tutti gli split
        self.splits = {}
        for split_name, split_data in self.dataset_splits.items():
            print('Splitting {}...'.format(split_name))
            self.splits[split_name] = split_data.map(preprocess_token, batched=True, remove_columns=['article', 'abstract'])
            print("Caching split {}...".format(split_name))
            self.splits[split_name].save_to_disk(DATA_DIR / f'{split_name}_tokenized')
        print("✅ Tokenizzazione completata per train/val/test")

    def train(self):
        """
            :return: tokenized train split
        """
        return self.splits['train']

    def eval(self):
        """
        :return: tokenized validation split
        """
        return self.splits['validation']

    def test(self):
        """
        :return: tokenized test split
        """
        return self.splits['test']

    def save_splits(self):
        # 7. Salvataggio CSV puliti
        for split_name, split_data in self.dataset_splits.items():
            split_df = pd.DataFrame(split_data)
            split_df.to_csv(DATA_DIR / f"{split_name}_clean.csv", index=False)
            print(f"✅ Salvato {split_name}_clean.csv con {len(split_df)} righe")




if __name__ == '__main__':
    p_test = Preprocessor()