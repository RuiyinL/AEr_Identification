<div align="center">
  <h1 align="center">Towards Automated Identification of Violation Symptoms of Architecture Erosion</h1>
</div>

<div align="center">
    <a href="https://github.com/RuiyinL/AEr_Identification">
        <img src="https://img.shields.io/badge/GitHub-000?logo=github&logoColor=FFE165&style=for-the-badge" alt="">
    </a>
    <a href="https://arxiv.org/abs/2306.08616">
        <img src="https://img.shields.io/badge/Paper-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="">
    </a>
    <hr>
</div>

This repository is the replication package for the paper *Towards Automated Identification of Violation Symptoms of Architecture Erosion*. It provides the labeled code review data, extracted word-embedding features, preprocessing and classification scripts, LLM prompt templates, and practitioner survey/interview materials used in the study.

## Overview

Architecture erosion is the gradual divergence between implemented software architecture and intended architectural design. The paper studies **violation symptoms** as early textual indicators of architecture erosion in code review discussions, and evaluates whether these symptoms can be identified automatically.

The original study manually examined **21,583 code review comments** from four open-source projects: **OpenStack Nova**, **OpenStack Neutron**, **Qt Base**, and **Qt Creator**. From this corpus, the authors identified **606 violation-symptom comments**. This replication package contains those labeled violation symptoms and a balanced set of **606 randomly selected non-violation comments** used in the ML/DL experiments.

The paper follows a three-phase workflow:

- **Phase 1: ML/DL-based classification.** The study preprocesses code review comments, extracts word-embedding features, trains ML and DL classifiers, and evaluates their performance on identifying violation symptoms.
- **Phase 2: Practitioner validation.** The study validates the usefulness of automatically identified violation symptoms through survey and interview materials, and reports a controlled experiment for comparison validation.
- **Phase 3: LLM-based comparison.** The study builds LLM-based classifiers through prompt engineering and compares them with the ML/DL approaches from Phase 1.

<img src="img/Overview.png" alt="Overview of the research process" style="zoom: 33%;" />

## Main Findings

The paper evaluates ML, DL, and LLM-based classifiers:

- **ML classifiers:** Support Vector Machine (SVM), Logistic Regression (LR), Decision Tree (DT), Bernoulli Naive Bayes (NB), and k-Nearest Neighbor (kNN).
- **DL classifiers:** TextCNN variants using a vocabulary-based representation and pre-trained embeddings.
- **LLM classifiers:** GPT-4o, Qwen-2.5, and DeepSeek-R1 in the balanced setting; the imbalanced-test-set analysis reported in the paper uses Qwen-3 instead of Qwen-2.5.

Among the ML/DL classifiers, **SVM achieves the best average F1-score of 0.779 across the three word embeddings**, and **SVM with StackOverflow word2vec achieves an F1-score of 0.808**. The paper also reports that **200-dimensional embeddings** generally outperform 100- and 300-dimensional variants. Among the LLM-based classifiers, **GPT-4o achieves the highest F1-score of 0.851** in the balanced setting.

## Package Contents

```plaintext
├── data/                         // labeled comments, extracted features, and word-embedding resources
│   ├── extracted_features/       // precomputed FastText, GloVe, and StackOverflow word2vec feature CSVs
│   └── word_embedding/           // embedding download links and embedding-dimension settings
├── img/                          // overview figure of the research process
├── scripts/                      // preprocessing and classification code
│   ├── preprocessing/            // database management, preprocessing, and feature-extraction scripts
│   └── classifiers/              // ML, DL, and LLM classifiers plus prompt templates
│       └── prompt_templates/     // LLM prompt templates
├── survey and interview/         // survey form, interview protocol, and customized email templates
├── controlled experiments        // controlled experiments protocol
└── README.md                     // package documentation
```

## Notes on Reuse

- The package provides the labeled violation-symptom dataset and the balanced non-violation sample used in the main ML/DL experiments; it does not include the full original set of 21,583 manually examined comments.
- Large pre-trained embedding files must be downloaded separately using the links in `data/word_embedding/Download_url.txt`.
- LLM experiments require configuring an API client and model names in `scripts/classifiers/LLM.py`.

## Citation

```bibtex
@article{Li2026ViolationSymptoms,
  author = {Li, Ruiyin and Liang, Peng and Avgeriou, Paris and Wang, Yifei},
  title = {{Towards Automated Identification of Violation Symptoms of Architecture Erosion}},
  journal = {arXiv preprint arXiv:2306.08616},
  year = {2026}
}
```
