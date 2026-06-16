import json
import pandas as pd

# the random seed you set in the run.py
RANDOM_SEED = 42
models = ["gpt-4o", "qwen-plus", "deepseek-reasoner"]
templates = ["template_1"]
shots = [0,2,4,6,10]

with open("./dataset/Violation_symptoms.xlsx", mode='rb') as f:
    violation_frame = pd.read_excel(f)

with open("./dataset/Randomly_selected_comments.xlsx", mode='rb') as f:
    non_violation_frame = pd.read_excel(f)

violations = violation_frame["Comment"].tolist()
non_violation = non_violation_frame["Comment"].tolist()

def check_violation(comment):
    if comment in violations:
        return True
    if comment in non_violation:
        return False
    
    # error here
    print("NOTICE: error comment args passed in.")
    exit()

def calculate_metrics(comments):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for comment in comments:
        pred = comments[comment]
        true = check_violation(comment)
        if true == True and pred == True:
            TP += 1
        elif true == False and pred == True:
            FP += 1
        elif true == True and pred == False:
            FN += 1
        else:
            TN += 1
    
    acc = (TP + TN) / (len(comments))
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    print(TP, FP, FN, TN)
    return f1, recall, precision, acc

for model in models:
    target_json = f"./out/{str(RANDOM_SEED)}/results-{model}.json"
    with open(target_json, mode='r', encoding='utf-8') as f:
        json_content = json.load(f)
    for template in templates:
        for shot in shots:
            result = json_content[template][str(shot)]
            f1, r, p, acc = calculate_metrics(result)
            print(f"{model}-{template}-{str(shot)}-shot:\nF1: {str(f1)}\nRecall: {str(r)}\nPrecision: {str(p)}\nAccuracy: {str(acc)}\n")

