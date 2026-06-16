import os
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import trange
from sklearn import metrics
from torchtext import vocab
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchtext.transforms import VocabTransform
from torchtext.data.utils import ngrams_iterator
import matplotlib.pyplot as plt
from preprocessing.preprocessing import preprocess
from classifiers.DL_models import parse_args
from classifiers.DL_models import TextCNN_voc, TextCNN_w2v, TextCNN_fastText, TextCNN_GloVe
from classifiers.DL_utility import (EarlyStopping, dataset_split, pad_or_cut,
                                     generate_tensor, load_test_data,
                                     CBFocalLoss, TemperatureScaling,
                                     apply_smote_to_training_data, SMOTEDataset,
                                     predict_with_threshold, compute_macro_f1)

# torch.cuda.empty_cache()        # release memory
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, text_list, text_label, args):
        '''
        :param text_list: sentence_list
        :param text_label: sentence_label
        :param args: load max_length
        '''
        super(MyDataset, self).__init__()
        self.args = args
        text_vocab, vocab_transform = self.build_vocab(text_list)
        self.text_list = text_list
        self.text_label = text_label
        self.text_vocab = text_vocab
        self.vocab_transform = vocab_transform
        self.data = self.generate_data()
        self.size = self.get_vocab_size()
        self.max_length = self.args.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id_index):
        token = self.data[id_index]
        label = self.text_label[id_index]
        return token, label

    # Build a vocabulary
    def build_vocab(self, text_list):   # sentence_list = ['xx',..., 'xxx']
        total_word_list = []
        for sentence in text_list:
            sentence = preprocess(sentence)
            total_word_list += list(ngrams_iterator(sentence, 2))  # n-gram
        counter = Counter(total_word_list)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        special_token = ['<UNK>', '<SEP>']
        text_vocab = vocab(ordered_dict, specials=special_token)
        text_vocab.set_default_index(0)
        vocab_transform = VocabTransform(text_vocab)
        return text_vocab, vocab_transform

    def generate_data(self):
        token_id_list = []
        for sentence in self.text_list:
            sentence_words = preprocess(sentence)
            sentence_id_list = np.array(self.vocab_transform(sentence_words))
            sentence_id_list = pad_or_cut(sentence_id_list, self.args.max_length)
            token_id_list.append(sentence_id_list)
        return token_id_list

    def get_vocab_size(self):
        return len(self.text_vocab)


class MyDataset_pre_trained(Dataset):
    def __init__(self, text_list, text_label, args):
        super(MyDataset_pre_trained, self).__init__()
        self.args = args
        self.x = text_list
        self.y = text_label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, id_index):
        label = self.y[id_index]
        sentence = self.x[id_index]
        word_list = preprocess(sentence)
        word_list = np.array(word_list)
        word_list = pad_or_cut(word_list, self.args.max_length)
        if self.args.TextCNN_w2v:
            sentence_vector = generate_tensor('SO_w2v', word_list, len(word_list), self.args)
        if self.args.TextCNN_fastText:
            # sentence_vector = generate_tensor('FastText_200', word_list, len(word_list), args)  # return <class 'torch.Tensor'>
            sentence_vector = generate_tensor('FastText', word_list, len(word_list), self.args)
        if self.args.TextCNN_GloVe:
            sentence_vector = generate_tensor('GloVe', word_list, len(word_list), self.args)
        return sentence_vector, label


def build_class_weights(labels, pos_weight=1.0, neg_weight=1.0):
    weight_neg = float(neg_weight)
    weight_pos = float(pos_weight)
    return torch.tensor([weight_neg, weight_pos], dtype=torch.float)


def train(args, model, train_iter, valid_iter, class_weights=None, criterion=None):
    time_start = time.time()
    if torch.cuda.is_available():
        model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # ============= optimizer =============
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # ============= Loss function =============
    if criterion is None:
        if class_weights is not None:
            class_weights = class_weights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
    else:
        criterion = criterion.to(DEVICE)
    # ============= Early Stopping =============
    early_stopping = EarlyStopping(patience=8, verbose=True)

    steps = 0
    total_loss = 0.
    global_step = 0
    train_epoch = trange(args.epoch, colour='blue', desc='train_epoch')
    train_losses = []
    valid_losses = []

    # ============= Training =============
    for epoch in train_epoch:
        model.train()

        for text_token, text_label in train_iter:
            global_step += 1
            optimizer.zero_grad()
            # model_out = model(text_token.to(DEVICE, non_blocking=True))
            # loss = criterion(model_out, text_label.to(DEVICE, non_blocking=True))
            model_out = model(text_token.to(DEVICE))
            loss = criterion(model_out, text_label.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        # print("Training:", epoch_i + 1, "Loss:", np.sum(loss_list))
        training_loss = total_loss / global_step
        print("Epoch:", epoch + 1, "Training loss:", training_loss)
        train_losses.append(training_loss)
        # scheduler.step()

        '''checkpoint'''
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "steps": steps,
                          # "training_loss": training_loss,
                          "global_step": global_step,
                          "total_loss": total_loss}
            path_checkpoint = "checkpoints/checkpoint_{}_epoch.pkl".format(epoch + 1)
            torch.save(checkpoint, path_checkpoint)


        '''Validation'''
        valid_loss, valid_result = evaluate(criterion, model, valid_iter)
        valid_losses.append(valid_loss)
        # Early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break

    torch.cuda.empty_cache()    # dump GPU cache
    time_end = time.time()
    train_time = time_end - time_start

    # torch.save(model.state_dict(), 'checkpoints/My_model.pth')
    torch.save(model, 'checkpoints/My_model.pth')

    '''loss vasualization'''
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.ylim(ymin=0, ymax=1.01)
    plt.title("The loss of current model")
    plt.legend(["train loss", 'validation loss'])
    plt.show()
    print('Train time:', train_time, 's')


def evaluate(criterion, model, valid_test_iter, threshold=None,
              reject_interval=None):
    model.eval()
    total_loss = 0.
    total_step = 0.
    preds = None
    true_label = None
    pred_scores = None
    with torch.no_grad():
        for text_token, text_label in valid_test_iter:
            model_out = model(text_token.to(DEVICE))
            example_losses = criterion(model_out, text_label.to(DEVICE))
            total_loss += example_losses.item()
            total_step += 1
            probs = torch.softmax(model_out, dim=1).detach().cpu().numpy()

            if preds is None:
                true_label = text_label.detach().cpu().numpy()
                pred_scores = probs[:, 1]
            else:
                true_label = np.append(true_label, text_label.detach().cpu().numpy(), axis=0)
                pred_scores = np.append(pred_scores, probs[:, 1], axis=0)

    # Decision policy: threshold + optional reject interval
    if threshold is not None:
        preds = predict_with_threshold(pred_scores, threshold=threshold,
                                       reject_interval=reject_interval)
    else:
        preds = (pred_scores >= 0.5).astype(int)

    print("Predicted labels:", preds)
    result = acc_and_f1(preds, true_label, pred_scores)

    return total_loss / total_step, result


'''Performance'''
def acc_and_f1(preds, Y_test, pred_scores=None):
    # acc = (preds == Y_test).mean()
    acc = metrics.accuracy_score(y_true=Y_test, y_pred=preds)
    f1 = metrics.f1_score(y_true=Y_test, y_pred=preds, average='weighted')
    precision = metrics.precision_score(y_true=Y_test, y_pred=preds)
    recall = metrics.recall_score(y_true=Y_test, y_pred=preds)
    macro_f1 = compute_macro_f1(Y_test, preds)
    auc_pr = metrics.average_precision_score(Y_test, pred_scores) if pred_scores is not None else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "auc_pr": auc_pr,
    }


def load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, args):
    if args.TextCNN_voc:
        train_data = MyDataset(X_train, Y_train, args)
        # vocab_size = train_data.size
        valid_data = MyDataset(X_valid, Y_valid, args)
        test_data = MyDataset(X_test, Y_test, args)

    if args.TextCNN_w2v or args.TextCNN_fastText or args.TextCNN_GloVe:
        train_data = MyDataset_pre_trained(X_train, Y_train, args)
        valid_data = MyDataset_pre_trained(X_valid, Y_valid, args)
        test_data = MyDataset_pre_trained(X_test, Y_test, args)

    train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_iter = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    return train_iter, valid_iter, test_iter


def run_textcnn_experiment(args, model_type, class_weights=None,
                           imbalanced_config=None):
    """Run a single TextCNN experiment (train + calibration + test).

    Parameters
    ----------
    args : Namespace
    model_type : str
    class_weights : torch.Tensor or None
        Used only when ``imbalanced_config`` is None (legacy path).
    imbalanced_config : dict or None
        When provided, enables the full cost‑sensitive pipeline
        (Table 10 of the paper).  Expected keys:
          - cb_focal_params : dict  (samples_per_class, beta, gamma,
                                     alpha_pos, pi_1)
          - smote_params    : dict  (target_pos_ratio, random_state)
          - cal_params      : dict  (lr, max_iter)
          - threshold       : float
          - reject_interval : tuple or None
    """
    X_train, Y_train, X_valid, Y_valid, _, _ = dataset_split(
        args,
        label_path0=getattr(args, "label_path0", None) or None,
        label_path1=getattr(args, "label_path1", None) or None,
        test_size=getattr(args, "test_size", 1 / 5),
        valid_size=getattr(args, "valid_size", 1 / 4),
        random_seed=getattr(args, "random_seed", 6),
        n_per_class=getattr(args, "n_per_class", 599),
    )
    X_test, Y_test = load_test_data(
        test_path=getattr(args, "test_path", "~/experiment/data/Test set.xlsx"),
        sheet_name=getattr(args, "test_sheet", "combination"),
    )

    # ---------- SMOTE (training partition only) ----------
    if imbalanced_config is not None and imbalanced_config.get('smote_params'):
        smote_cfg = imbalanced_config['smote_params']
        X_train_tensors, Y_train_aug = apply_smote_to_training_data(
            X_train, Y_train, args, **smote_cfg
        )
        train_dataset = SMOTEDataset(X_train_tensors, Y_train_aug)
        train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # valid/test are never resampled — get them from load_dataset
        _, valid_iter, test_iter = load_dataset(
            X_train, Y_train, X_valid, Y_valid, X_test, Y_test, args
        )
    else:
        # Standard path — all three from load_dataset
        train_iter, valid_iter, test_iter = load_dataset(
            X_train, Y_train, X_valid, Y_valid, X_test, Y_test, args
        )

    # ---------- model ----------
    if model_type == "TextCNN_voc":
        train_data = MyDataset(X_train, Y_train, args)
        vocab_size = train_data.size
        model = TextCNN_voc(args, vocab_size).to(DEVICE)
    elif model_type == "TextCNN_w2v":
        model = TextCNN_w2v(args).to(DEVICE)
    elif model_type == "TextCNN_fastText":
        model = TextCNN_fastText(args).to(DEVICE)
    elif model_type == "TextCNN_GloVe":
        model = TextCNN_GloVe(args).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ---------- loss ----------
    if imbalanced_config is not None:
        cb_cfg = imbalanced_config.get('cb_focal_params', {})
        criterion = CBFocalLoss(**cb_cfg)
        train(args, model, train_iter, valid_iter, criterion=criterion)
    else:
        criterion = None
        train(args, model, train_iter, valid_iter, class_weights=class_weights)

    # ---------- calibration (temperature scaling) ----------
    if imbalanced_config is not None and imbalanced_config.get('cal_params'):
        # Gather validation logits
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for text_token, text_label in valid_iter:
                logits = model(text_token.to(DEVICE)).detach().cpu()
                val_logits_list.append(logits)
                val_labels_list.append(text_label)
        val_logits = torch.cat(val_logits_list, dim=0)
        val_labels = torch.cat(val_labels_list, dim=0)

        scaler = TemperatureScaling()
        temperature = scaler.calibrate(val_logits, val_labels,
                                       **imbalanced_config['cal_params'])
        print(f"[Temperature Scaling] learned T = {temperature:.4f}")

        # Build calibrated criterion for test evaluation
        if criterion is None:
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
        test_criterion = criterion
        # Wrap model with temperature scaling for test
        original_forward = model.forward

        class CalibratedModel(nn.Module):
            def __init__(self, base_model, temperature_module):
                super().__init__()
                self.base = base_model
                self.scaler = temperature_module

            def forward(self, x):
                logits = self.base(x)
                return self.scaler(logits)

        cal_model = CalibratedModel(model, scaler).to(DEVICE)
        threshold = imbalanced_config.get('threshold', 0.65)
        reject_interval = imbalanced_config.get('reject_interval', (0.45, 0.55))
        test_loss, test_result = evaluate(test_criterion, cal_model, test_iter,
                                          threshold=threshold,
                                          reject_interval=reject_interval)
        return test_loss, test_result

    # ---------- legacy evaluation ----------
    if criterion is None:
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
    test_loss, test_result = evaluate(criterion, model, test_iter)
    return test_loss, test_result


def cross_validate_textcnn(args, model_type, class_weights=None,
                           imbalanced_config=None):
    from sklearn.model_selection import StratifiedKFold
    X_train, Y_train, _, _, _, _ = dataset_split(
        args,
        label_path0=getattr(args, "label_path0", None) or None,
        label_path1=getattr(args, "label_path1", None) or None,
        test_size=getattr(args, "test_size", 1 / 5),
        valid_size=getattr(args, "valid_size", 1 / 4),
        random_seed=getattr(args, "random_seed", 6),
        n_per_class=getattr(args, "n_per_class", 599),
    )
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    skf = StratifiedKFold(
        n_splits=getattr(args, "cv_folds", 10),
        shuffle=True,
        random_state=getattr(args, "random_seed", 6),
    )
    fold_metrics = []
    for train_idx, val_idx in skf.split(X_train, Y_train):
        X_tr = X_train[train_idx].tolist()
        Y_tr = Y_train[train_idx].tolist()
        X_val = X_train[val_idx].tolist()
        Y_val = Y_train[val_idx].tolist()

        # SMOTE for this fold (training partition only)
        if imbalanced_config is not None and imbalanced_config.get('smote_params'):
            smote_cfg = imbalanced_config['smote_params']
            X_tr_tensors, Y_tr_aug = apply_smote_to_training_data(
                X_tr, Y_tr, args, **smote_cfg
            )
            train_dataset = SMOTEDataset(X_tr_tensors, Y_tr_aug)
            train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            _, valid_iter, _ = load_dataset(
                X_tr, Y_tr, X_val, Y_val, X_val, Y_val, args
            )
        else:
            train_iter, valid_iter, _ = load_dataset(
                X_tr, Y_tr, X_val, Y_val, X_val, Y_val, args
            )

        if model_type == "TextCNN_voc":
            train_data = MyDataset(X_tr, Y_tr, args)
            vocab_size = train_data.size
            model = TextCNN_voc(args, vocab_size).to(DEVICE)
        elif model_type == "TextCNN_w2v":
            model = TextCNN_w2v(args).to(DEVICE)
        elif model_type == "TextCNN_fastText":
            model = TextCNN_fastText(args).to(DEVICE)
        elif model_type == "TextCNN_GloVe":
            model = TextCNN_GloVe(args).to(DEVICE)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Loss
        if imbalanced_config is not None:
            cb_cfg = imbalanced_config.get('cb_focal_params', {})
            criterion = CBFocalLoss(**cb_cfg)
            train(args, model, train_iter, valid_iter, criterion=criterion)
        else:
            train(args, model, train_iter, valid_iter, class_weights=class_weights)
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

        # Calibration on validation fold
        if imbalanced_config is not None and imbalanced_config.get('cal_params'):
            model.eval()
            val_logits_list, val_labels_list = [], []
            with torch.no_grad():
                for text_token, text_label in valid_iter:
                    logits = model(text_token.to(DEVICE)).detach().cpu()
                    val_logits_list.append(logits)
                    val_labels_list.append(text_label)
            val_logits = torch.cat(val_logits_list, dim=0)
            val_labels = torch.cat(val_labels_list, dim=0)

            scaler = TemperatureScaling()
            scaler.calibrate(val_logits, val_labels,
                             **imbalanced_config['cal_params'])
            # Re-evaluate with temperature scaling
            threshold = imbalanced_config.get('threshold', 0.65)
            reject_interval = imbalanced_config.get('reject_interval', (0.45, 0.55))
            model.eval()
            all_probs = []
            with torch.no_grad():
                for text_token, _ in valid_iter:
                    logits = model(text_token.to(DEVICE)).detach().cpu()
                    cal_logits = scaler(logits)
                    probs = torch.softmax(cal_logits, dim=1)
                    all_probs.append(probs)
            all_probs = torch.cat(all_probs, dim=0).numpy()
            all_preds = predict_with_threshold(all_probs[:, 1],
                                               threshold=threshold,
                                               reject_interval=reject_interval)
            fold_result = acc_and_f1(all_preds, Y_val, all_probs[:, 1])
        else:
            _, fold_result = evaluate(criterion, model, valid_iter)

        fold_metrics.append(fold_result)

    return fold_metrics


if __name__ == '__main__':
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        DEVICE = torch.device("cpu")
    global args
    args = parse_args()
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = dataset_split(args)
    class_weights = build_class_weights(Y_train, pos_weight=1.0, neg_weight=1.0).to(DEVICE)


    '''Training'''
    if args.do_train:
        if args.TextCNN_voc:
            print('=====Train TextCNN_voc=====')
            train_data = MyDataset(X_train, Y_train, args)
            vocab_size = train_data.size
            train_iter, valid_iter, test_iter = load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, args)
            TextCNN_voc_model = TextCNN_voc(args, vocab_size).to(DEVICE)
            train(args, TextCNN_voc_model, train_iter, valid_iter, class_weights=class_weights)

        if args.TextCNN_w2v:
            print('=====Train TextCNN_w2v=====')
            train_iter, valid_iter, test_iter = load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, args)
            TextCNN_w2v_model = TextCNN_w2v(args).to(DEVICE)
            train(args, TextCNN_w2v_model, train_iter, valid_iter, class_weights=class_weights)

        if args.TextCNN_fastText:
            print('=====Train TextCNN_fastText=====')
            train_iter, valid_iter, test_iter = load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, args)
            TextCNN_fastText_model = TextCNN_fastText(args).to(DEVICE)
            train(args, TextCNN_fastText_model, train_iter, valid_iter, class_weights=class_weights)

        if args.TextCNN_GloVe:
            print('=====Train TextCNN_GloVe=====')
            train_iter, valid_iter, test_iter = load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, args)
            TextCNN_GloVe_model = TextCNN_GloVe(args).to(DEVICE)
            train(args, TextCNN_GloVe_model, train_iter, valid_iter, class_weights=class_weights)


    '''Testing'''
    if args.do_test:
        print("===== Start testing =====")
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
        if args.TextCNN_voc:
            print('=====Test TextCNN_voc=====')
            TextCNN_voc_model = TextCNN_voc(args, vocab_size).to(DEVICE)
            TextCNN_voc_test_loss, TextCNN_voc_result = evaluate(criterion, TextCNN_voc_model, test_iter)
            print("TextCNN_voc_test_loss", TextCNN_voc_test_loss)
            print("TextCNN_voc_result", TextCNN_voc_result)

        if args.TextCNN_w2v:
            print('=====Test TextCNN_w2v=====')
            TextCNN_w2v_model = TextCNN_w2v(args).to(DEVICE)
            TextCNN_w2v_test_loss, TextCNN_w2v_result = evaluate(criterion, TextCNN_w2v_model, test_iter)
            print("TextCNN_w2v_test_loss", TextCNN_w2v_test_loss)
            print("TextCNN_w2v_result", TextCNN_w2v_result)

        if args.TextCNN_fastText:
            print('=====Test TextCNN_fastText=====')
            TextCNN_fastText_model = TextCNN_fastText(args).to(DEVICE)
            TextCNN_fastText_test_loss, TextCNN_fastText_result = evaluate(criterion, TextCNN_fastText_model, test_iter)
            print("TextCNN_fastText_test_loss", TextCNN_fastText_test_loss)
            print("TextCNN_fastText_result", TextCNN_fastText_result)

        if args.TextCNN_GloVe:
            print('=====Test TextCNN_GloVe=====')
            TextCNN_GloVe_model = TextCNN_GloVe(args).to(DEVICE)
            TextCNN_GloVe_test_loss, TextCNN_GloVe_result = evaluate(criterion, TextCNN_GloVe_model, test_iter)
            print("TextCNN_GloVe_test_loss", TextCNN_GloVe_test_loss)
            print("TextCNN_GloVe_result", TextCNN_GloVe_result)
