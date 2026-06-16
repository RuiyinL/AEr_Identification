import math
import os
import torch
import gensim
import fasttext
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from preprocessing.w2vemb import EMB
from torchtext.data import get_tokenizer
from collections import Counter, OrderedDict
from gensim.models.keyedvectors import KeyedVectors
from torchtext.data.utils import ngrams_iterator
from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import preprocess, pad_or_cut
from w2vembeddings.managedb import ManageDB
from w2vembeddings.w2vemb import EMB

path0 = r'~/experiment/data/Randomly_selected_comments.xlsx'
path1 = r'~/experiment/data/Violation symptoms.xlsx'
path_w2v0 = r'~/experiment/data/extracted_features/SO_w2v_200_non_violation.csv'  # negative data (for TextCNN_w2v)
path_w2v1 = r'~/experiment/data/extracted_features/SO_w2v_200_violation.csv'      # positive data (for TextCNN_w2v)
# path_ft0 = r'~/experiment/data/extracted_features/FastText_200_non_violation.csv'  # negative data (for TextCNN_fastText)
# path_ft1 = r'~/experiment/data/extracted_features/FastText_200_violation.csv'      # positive data (for TextCNN_fastText)
path_ft0 = r'~/experiment/data/extracted_features/FastText_100_non_violation.csv'  # negative data (for TextCNN_fastText)
path_ft1 = r'~/experiment/data/extracted_features/FastText_100_violation.csv'      # positive data (for TextCNN_fastText)

# Preload labels for optional use in scripts that rely on module-level data.
label0 = pd.read_excel(os.path.expanduser(path0), sheet_name='Comments', na_values='n/a')
label1 = pd.read_excel(os.path.expanduser(path1), sheet_name='combination', na_values='n/a')


wv_from_bin = KeyedVectors.load_word2vec_format(
    os.path.expanduser("~/experiment/data/word_embedding/SO_vectors_200.bin"),
    binary=True,
)
fasttext.FastText.eprint = lambda x: None
ft = fasttext.load_model(os.path.expanduser('~/experiment/data/word_embedding/cc.en.100.bin'))
# ft = fasttext.load_model(os.path.expanduser('~/experiment/data/word_embedding/cc.en.200.bin'))
# ft = fasttext.load_model(os.path.expanduser('~/experiment/data/word_embedding/cc.en.300.bin'))
word2id = wv_from_bin.key_to_index  # dict: {word, index}; like this: {'a': 0, 'b', 1, ...}
ft_word_dic = ft.words              # vocabulary list; like this: ['the', 'design', ..., 'Zwicke']

# gensim_file = '~/experiment/data/word_embedding/glove.twitter.27B.200d.txt'
# md = ManageDB()
# md.add_file2db('glove.twitter.27B.200d', gensim_file, 200, 1193513)    # write it into database (only need to run in the first time)
GloVe_200 = EMB(name='glove.twitter.27B.200d', dimensions=200)

def dataset_split(
    args,
    label_path0=path0,
    label_path1=path1,
    test_size=1 / 5,
    valid_size=1 / 4,
    random_seed=6,
    n_per_class=599,
):
    '''
    split training set, validation set, test set,比例是6：2：2
    '''
    label_path0 = label_path0 or path0
    label_path1 = label_path1 or path1
    label0_df = pd.read_excel(os.path.expanduser(label_path0), sheet_name='Comments', na_values='n/a')
    label1_df = pd.read_excel(os.path.expanduser(label_path1), sheet_name='combination', na_values='n/a')
    non_violation = label0_df[['Comment', 'Label']].sample(n=n_per_class, random_state=random_seed)
    violation = label1_df[['Comment', 'Label']].sample(n=n_per_class, random_state=random_seed)
    x0 = non_violation['Comment'].tolist()
    x1 = violation['Comment'].tolist()

    y0 = non_violation['Label'].tolist()
    y1 = violation['Label'].tolist()
    X_train_valid0, X_test0, Y_train_valid0, Y_test0 = train_test_split(
        x0, y0, test_size=test_size, random_state=random_seed
    )
    X_train_valid1, X_test1, Y_train_valid1, Y_test1 = train_test_split(
        x1, y1, test_size=test_size, random_state=random_seed
    )
    X_train0, X_valid0, Y_train0, Y_valid0 = train_test_split(
        X_train_valid0, Y_train_valid0, test_size=valid_size, random_state=random_seed
    )
    X_train1, X_valid1, Y_train1, Y_valid1 = train_test_split(
        X_train_valid1, Y_train_valid1, test_size=valid_size, random_state=random_seed
    )

    X_train = X_train0 + X_train1  # list
    X_valid = X_valid0 + X_valid1
    X_test = X_test0 + X_test1
    Y_train = Y_train0 + Y_train1
    Y_valid = Y_valid0 + Y_valid1
    Y_test = Y_test0 + Y_test1
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def load_test_data(test_path="~/experiment/data/Test set.xlsx", sheet_name="combination"):
    test_df = pd.read_excel(os.path.expanduser(test_path), sheet_name=sheet_name, na_values='n/a')
    X_test = test_df['Comment'].tolist()
    Y_test = test_df['Label'].tolist()
    return X_test, Y_test


def pad_or_cut(value: np.array, target_length: int):  # value: np.ndarray, target_length: int
    # value = np.array(value)
    data_row = None
    if len(value) < target_length:
        data_row = np.pad(value, (0, target_length - len(value)), 'constant', constant_values=int(0))
    elif len(value) > target_length:
        data_row = value[:target_length]
    return data_row


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        '''
        :param patience (int): How long to wait after last time validation loss improved. Default: 6
        :param verbose (bool): If True, prints a message for each validation loss improvement. Default: False
        :param delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        :param path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
        :param trace_func (function): trace print function. Default: print
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def generate_tensor(embeding_model, sentence, _len, args):
    """transform to FloatTensor
    :param embeding_model: 'SO_w2v_200' or 'FastText_200'
    :param sentence: word list
    :param _len: sentence length
    :return: tensor
    """
    tensor = torch.zeros([args.max_length, args.embedding_dim])
    for index in range(0, args.max_length):
        if index >= len(sentence):
            break
        else:
            word = sentence[index]
            if embeding_model == 'SO_w2v':
                if word == '0':
                    word = '<UNK>'  # replace '0'
                if word in word2id:
                    # tensor[index] = wv_from_bin.get_vector(word)  # vector, <class 'numpy.ndarray'>
                    tensor[index] = torch.FloatTensor(wv_from_bin.get_vector(word))

            if embeding_model == 'FastText':
                if word == '0':
                    word = '<UNK>'  # replace '0'
                if word in ft_word_dic:
                    # tensor[index] = ft.get_word_vector(word)
                    tensor[index] = torch.FloatTensor(ft.get_word_vector(word))

            if embeding_model == 'GloVe':
                if word == '0':
                    word = '<UNK>'  # replace '0'
                if word in GloVe_200:
                    tensor[index] = torch.FloatTensor(np.array(GloVe_200.get_vector(word)))
    return tensor.unsqueeze(0)


def Save_Checkpoint(epoch, epochs_since_improvement, model, optimizer, loss):#, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    torch.save(state, filename)


# ============================================================================
# Imbalanced-setting components (Section 4.1.2 of the TOSEM paper)
# ============================================================================

class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss with explicit cost weight.

    Combines three mechanisms from the literature:
      - Focal modulation     (Lin et al. 2017, ICCV):  (1 − p_t)^γ
      - Class-balanced weight (Cui et al. 2019, CVPR): (1−β) / (1−β^{n_y})
      - Cost-sensitive weight:  α⁺ for the positive (violation) class

    Parameters (Table 10 of the paper)
    -----------------------------------
    samples_per_class : [n_neg, n_pos]
        Number of training samples in each class (used for CB re‑weighting).
    beta : float = 0.9999
        Class‑balanced hyper‑parameter (β → 1 ⇒ mild re‑weighting).
    gamma : float = 2.0
        Focal‑loss focusing parameter.
    alpha_pos : float = 6.0
        Explicit cost weight assigned to the positive (violation) class.
    pi_1 : float = 0.75
        Effective prior of the positive class.
        When pi_1 ≠ 0.5, a logit bias  log(pi_1 / (1−pi_1))  is added
        to the positive logit so the loss implicitly shifts the
        decision boundary towards higher minority‑class recall.
    reduction : str = 'mean'
        Loss reduction ('mean' | 'sum' | 'none').
    """

    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0,
                 alpha_pos=6.0, pi_1=0.75, reduction='mean'):
        super(CBFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.reduction = reduction

        # --- class-balanced weights (Cui et al. 2019) ---
        effective_num = [(1.0 - beta ** n) / (1.0 - beta)
                         if n > 0 else 0.0
                         for n in samples_per_class]
        cb_weight = [1.0 / en if en > 0 else 0.0 for en in effective_num]
        # Normalise so the negative‑class weight equals 1.0
        if cb_weight[0] > 0:
            cb_weight = [w / cb_weight[0] for w in cb_weight]
        self.register_buffer('cb_weight', torch.tensor(cb_weight, dtype=torch.float))

        # --- prior‑based logit bias ---
        if pi_1 is not None and pi_1 > 0.0 and pi_1 < 1.0 and pi_1 != 0.5:
            self.logit_bias = math.log(pi_1 / (1.0 - pi_1))
        else:
            self.logit_bias = 0.0

    def forward(self, logits, targets):
        # 1. prior adjustment on the logits (optional)
        if self.logit_bias != 0.0:
            logits = logits + torch.tensor([0.0, self.logit_bias],
                                           device=logits.device)

        # 2. focal loss (per‑sample, no reduction)
        ce = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce)                     # p_t = exp(−CE)  ≡ model's conf. for the true class
        focal = (1.0 - p_t) ** self.gamma * ce

        # 3. class‑balanced weight
        cb = self.cb_weight.to(logits.device)[targets]

        # 4. cost‑sensitive α (positive class weight)
        alpha = torch.where(targets == 1,
                            torch.tensor(self.alpha_pos, device=logits.device),
                            torch.tensor(1.0, device=logits.device))

        loss = cb * alpha * focal

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TemperatureScaling(nn.Module):
    """Post‑hoc temperature calibration (Guo et al. 2017, ICML).

    Learns a single scalar temperature  T  on a held‑out validation set
    by minimising NLL.  The calibrated probability is
        p̂ = softmax(logit / T).

    Usage
    -----
        scaler = TemperatureScaling()
        temperature = scaler.calibrate(val_logits, val_labels)
        cal_logits = scaler(test_logits)         # logits / T
    """

    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits, labels, lr=0.01, max_iter=100):
        """Fit T on validation logits.

        Parameters
        ----------
        logits : Tensor (N, C) – model outputs *before* softmax.
        labels : Tensor (N,)  – ground‑truth class indices.
        lr : float = 0.01    – learning rate for L‑BFGS.
        max_iter : int = 100 – maximum L‑BFGS iterations.

        Returns
        -------
        temperature : float
        """
        self.train()
        device = logits.device if isinstance(logits, torch.Tensor) else next(self.parameters()).device
        logits = logits.to(device) if isinstance(logits, torch.Tensor) else torch.tensor(logits, device=device)
        labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)

        self.to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()
        return self.temperature.item()


# ---------------------------------------------------------------------------
# SMOTE oversampling for TextCNN embedding tensors
# ---------------------------------------------------------------------------

def _build_embedding_tensor(text_list, args):
    """Convert a list of raw texts into a 2‑D feature matrix.

    Each text is turned into a flat vector of length
    ``max_length * embedding_dim`` by running the same embedding
    procedure used in ``MyDataset_pre_trained``.
    """
    vectors = []
    for sentence in text_list:
        word_list = np.array(preprocess(sentence))
        word_list = pad_or_cut(word_list, args.max_length)
        if getattr(args, 'TextCNN_voc', False):
            # token IDs → one‑hot → flat
            ids = pad_or_cut(word_list, args.max_length)
            vec = np.eye(args.embedding_dim)[ids.astype(int)].flatten()
        elif getattr(args, 'TextCNN_w2v', False):
            tensor = torch.zeros(args.max_length, args.embedding_dim)
            for i, w in enumerate(word_list):
                if i >= args.max_length:
                    break
                if w == '0':
                    w = '<UNK>'
                if w in word2id:
                    tensor[i] = torch.FloatTensor(wv_from_bin.get_vector(w))
            vec = tensor.numpy().flatten()
        elif getattr(args, 'TextCNN_fastText', False):
            tensor = torch.zeros(args.max_length, args.embedding_dim)
            for i, w in enumerate(word_list):
                if i >= args.max_length:
                    break
                if w == '0':
                    w = '<UNK>'
                if w in ft_word_dic:
                    tensor[i] = torch.FloatTensor(ft.get_word_vector(w))
            vec = tensor.numpy().flatten()
        elif getattr(args, 'TextCNN_GloVe', False):
            tensor = torch.zeros(args.max_length, args.embedding_dim)
            for i, w in enumerate(word_list):
                if i >= args.max_length:
                    break
                if w == '0':
                    w = '<UNK>'
                if w in GloVe_200:
                    tensor[i] = torch.FloatTensor(np.array(GloVe_200.get_vector(w)))
            vec = tensor.numpy().flatten()
        else:
            raise ValueError("One of TextCNN_voc/w2v/fastText/GloVe must be True in args.")
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def apply_smote_to_training_data(X_train_texts, Y_train, args,
                                 target_pos_ratio=0.20, random_state=42):
    """SMOTE‑based oversampling for the TextCNN training partition.

    Flattens each sample's embedding tensor into a 2‑D feature matrix,
    applies SMOTE, and returns augmented lists.

    Parameters
    ----------
    X_train_texts : list[str]
        Raw training sentences.
    Y_train : list[int]
        Binary labels (0 = non‑violation, 1 = violation).
    args : Namespace
        Must carry TextCNN_voc / w2v / fastText / GloVe flags as well as
        max_length and embedding_dim.
    target_pos_ratio : float = 0.20
        Desired positive‑class fraction *after* SMOTE (ρ⁺ in the paper).
    random_state : int = 42

    Returns
    -------
    X_aug : list[torch.Tensor]   – each has shape (1, max_length, emb_dim)
    Y_aug : list[int]
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn is required for SMOTE. "
                          "Install with: pip install imbalanced-learn")

    X_flat = _build_embedding_tensor(X_train_texts, args)
    Y_arr = np.array(Y_train, dtype=int)

    n_neg = int(np.sum(Y_arr == 0))
    n_pos = int(np.sum(Y_arr == 1))

    # Determine sampling strategy to approach target_pos_ratio.
    # SMOTE can only *add* samples, so we oversample whichever class
    # is underrepresented relative to the target.
    total_target = n_neg + n_pos
    target_n_pos = int(target_pos_ratio * total_target / (1.0 - target_pos_ratio + target_pos_ratio))
    # simpler: ratio = pos / neg  →  pos = ratio * neg
    # but we also need to keep neg unchanged when reducing pos is impossible.
    if target_n_pos > n_pos:
        # need more positives → SMOTE typical use
        sampling_strategy = {1: target_n_pos, 0: n_neg}
    elif target_n_pos < n_pos:
        # need fewer positives → can't do with SMOTE alone;
        # oversample negatives instead to dilute the ratio.
        target_n_neg = int(n_pos / target_pos_ratio - n_pos)
        sampling_strategy = {0: target_n_neg, 1: n_pos}
    else:
        sampling_strategy = 'auto'

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, Y_res = smote.fit_resample(X_flat, Y_arr)

    # Reshape back to embedding‑tensor format expected by TextCNN
    max_len = args.max_length
    emb_dim = args.embedding_dim
    X_aug = []
    for row in X_res:
        tensor = torch.FloatTensor(row.reshape(1, max_len, emb_dim))
        X_aug.append(tensor)

    return X_aug, Y_res.tolist()


class SMOTEDataset(Dataset):
    """A thin ``Dataset`` wrapper around pre‑computed SMOTE‑augmented tensors."""

    def __init__(self, tensors, labels):
        self.tensors = tensors   # list of torch.Tensor
        self.labels = labels     # list[int]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Threshold‑based prediction & Macro‑F1 helpers
# ---------------------------------------------------------------------------

def predict_with_threshold(proba, threshold=0.65,
                           reject_interval=None):
    """Convert positive‑class probabilities to discrete predictions.

    Parameters
    ----------
    proba : np.ndarray  – P(y=1 | x), shape (N,)
    threshold : float = 0.65
        Decision threshold τ (Table 10).
    reject_interval : tuple | None = (0.45, 0.55)
        If not None, predictions falling inside (low, high] are
        mapped to −1 (abstain / reject).

    Returns
    -------
    preds : np.ndarray (int)  – 0, 1, or −1 (reject)
    """
    preds = (proba >= threshold).astype(int)
    if reject_interval is not None:
        low, high = reject_interval
        reject_mask = (proba > low) & (proba <= high)
        preds[reject_mask] = -1
    return preds


def compute_macro_f1(y_true, y_pred):
    """Unweighted mean of per‑class F1 scores (Macro‑F1).

    This is the primary evaluation metric for the imbalanced setting
    (Table 13 of the paper).
    """
    from sklearn.metrics import f1_score
    mask = y_pred != -1  # exclude rejected samples
    return f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
