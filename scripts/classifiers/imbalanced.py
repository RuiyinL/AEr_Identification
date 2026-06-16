import itertools
import json
import os
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from classifiers.ML_classifiers import run_ml_experiment
from classifiers.DL_classifiers import run_textcnn_experiment, build_class_weights, cross_validate_textcnn
from classifiers.DL_models import parse_args


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass
class MLConfig:
    model_name: str
    text_features: bool = True
    embeding_model: str = "FastText"
    add_stats: bool = True
    class_weight: Optional[dict] = None
    cost_matrix: Optional[list] = None
    dt_threshold: float = 0.5
    knn_threshold: float = 0.3
    knn_class_weights: Optional[dict] = None
    decision_threshold: float = 0.75
    oversample_method: Optional[str] = None
    cv: int = 5
    param_grid: Optional[dict] = None
    feature_path0: Optional[str] = None
    feature_path1: Optional[str] = None
    label_path0: Optional[str] = None
    label_path1: Optional[str] = None
    random_seed: int = 5
    n_per_class: int = 599
    test_path: str = None
    test_sheet: str = "combination"
    cv_folds: int = 10


@dataclass
class DLConfig:
    model_type: str
    pos_weight: float = 1.0
    neg_weight: float = 1.0
    lr: float = 1e-3
    batch_size: int = 16
    epoch: int = 50
    embedding_dim: int = 100
    max_length: int = 2000
    dropout: float = 0.5
    label_num: int = 2
    filter_sizes: str = "3,4,5"
    out_channels: int = 100
    checkpoint_interval: int = 1
    do_train: bool = True
    do_test: bool = True
    TextCNN_voc: bool = False
    TextCNN_w2v: bool = False
    TextCNN_fastText: bool = False
    TextCNN_GloVe: bool = True
    label_path0: str = "~/experiment/data/Randomly_selected_comments.xlsx"
    label_path1: str = "~/experiment/data/Violation symptoms.xlsx"
    test_size: float = 0.2
    valid_size: float = 0.25
    random_seed: int = 6
    n_per_class: int = 599
    test_path: str = None
    test_sheet: str = "combination"
    cv_folds: int = 10
    # --- imbalanced cost-sensitive pipeline (Table 10 of the paper) ---
    imbalanced_config: Optional[dict] = None


def run_ml_grid(configs):
    results = []
    for cfg in configs:
        metrics_out = run_ml_experiment(**asdict(cfg))
        results.append({"config": asdict(cfg), "metrics": metrics_out})
    return results


def run_dl_grid(configs):
    results = []
    for cfg in configs:
        args = parse_args()
        args.lr = cfg.lr
        args.batch_size = cfg.batch_size
        args.epoch = cfg.epoch
        args.embedding_dim = cfg.embedding_dim
        args.max_length = cfg.max_length
        args.dropout = cfg.dropout
        args.label_num = cfg.label_num
        args.filter_sizes = cfg.filter_sizes
        args.out_channels = cfg.out_channels
        args.checkpoint_interval = cfg.checkpoint_interval
        args.do_train = cfg.do_train
        args.do_test = cfg.do_test
        args.TextCNN_voc = cfg.TextCNN_voc
        args.TextCNN_w2v = cfg.TextCNN_w2v
        args.TextCNN_fastText = cfg.TextCNN_fastText
        args.TextCNN_GloVe = cfg.TextCNN_GloVe

        class_weights = build_class_weights([0, 1], pos_weight=cfg.pos_weight, neg_weight=cfg.neg_weight)
        args.label_path0 = cfg.label_path0
        args.label_path1 = cfg.label_path1
        args.test_size = cfg.test_size
        args.valid_size = cfg.valid_size
        args.random_seed = cfg.random_seed
        args.n_per_class = cfg.n_per_class
        args.test_path = cfg.test_path
        args.test_sheet = cfg.test_sheet

        fold_metrics = cross_validate_textcnn(
            args,
            cfg.model_type,
            class_weights=class_weights,
            imbalanced_config=cfg.imbalanced_config,
        )
        test_loss, test_result = run_textcnn_experiment(
            args,
            cfg.model_type,
            class_weights=class_weights,
            imbalanced_config=cfg.imbalanced_config,
        )
        results.append(
            {
                "config": asdict(cfg),
                "fold_metrics": fold_metrics,
                "test_loss": test_loss,
                "test_result": test_result,
            }
        )
    return results


def main():
    experiment_seed = 42  # set your seed
    set_global_seed(experiment_seed)

    # --- ML configs ---
    ml_configs = [
        MLConfig(model_name="SVM", class_weight={0: 1.0, 1: 10.0}, decision_threshold=0.70, oversample_method="smote", random_seed=experiment_seed),
        MLConfig(model_name="LR", class_weight={0: 1.0, 1: 8.0}, decision_threshold=0.65, oversample_method="smote", random_seed=experiment_seed),
        # cost_matrix[true_label][predicted_label]: FN cost (1 -> 0) is higher than FP cost (0 -> 1).
        MLConfig(model_name="NB", cost_matrix=[[0.0, 1.0], [10.0, 0.0]], oversample_method=None, random_seed=experiment_seed),
        MLConfig(model_name="DT", class_weight={0: 1.0, 1: 6.0}, dt_threshold=0.60, oversample_method=None, random_seed=experiment_seed),
        MLConfig(model_name="KNN", knn_class_weights={0: 1.0, 1: 5.0}, knn_threshold=0.55, oversample_method=None, random_seed=experiment_seed),
    ]

    # --- DL imbalanced config (Table 10 of the paper) ---
    imbalanced_config = {
        'cb_focal_params': {
            'samples_per_class': [599, 599],
            'beta': 0.9999,
            'gamma': 2.0,
            'alpha_pos': 6.0,
            'pi_1': 0.75,
        },
        'smote_params': {
            'target_pos_ratio': 0.20,
            'random_state': experiment_seed,
        },
        'cal_params': {
            'lr': 0.01,
            'max_iter': 100,
        },
        'threshold': 0.65,
        'reject_interval': (0.45, 0.55),
    }

    dl_configs = [
        DLConfig(model_type="TextCNN_GloVe", TextCNN_GloVe=True,
                 imbalanced_config=imbalanced_config, random_seed=experiment_seed),
        DLConfig(model_type="TextCNN_fastText", TextCNN_fastText=True,
                 TextCNN_GloVe=False,
                 imbalanced_config=imbalanced_config, random_seed=experiment_seed),
    ]
    ml_results = run_ml_grid(ml_configs)
    dl_results = run_dl_grid(dl_configs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"experiment_results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"ml": ml_results, "dl": dl_results}, f, indent=2)

    print(f"Experiment finished. Results saved to {output_path}")


if __name__ == "__main__":
    main()
