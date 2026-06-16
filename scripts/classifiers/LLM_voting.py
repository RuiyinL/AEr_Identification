import json
import pandas as pd

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

# the results
result_files = ["./out/42/results-deepseek-reasoner.json",
                "./out/42/results-gpt-4o.json",
                "./out/42/results-qwen-plus.json"]

results = {}
for file in result_files:
    with open(file, mode='r', encoding='utf-8') as f:
        result = json.load(f) 
    results[file] = result

ds = results[result_files[0]]["template_1"]
gpt = results[result_files[1]]["template_1"]
qwen = results[result_files[2]]["template_1"]

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
    
    return f1, recall, precision, acc

# votes is a list of True and False
def vote(votes):
    count_True = 0
    count_False = 0
    for ticket in votes:
        if ticket:
            count_True += 1
        elif not ticket:
            count_False += 1
        else:
            print("Error input list!")
            exit()

    if count_True > count_False:
        return True
    elif count_False > count_True:
        return False
    else:
        return "equal votes"


shots = ["0","2","4","6","10"]
models = ['deepseek-r1', 'gpt-4o', 'qwen-2.5']

def scenario_1():
    comments = ds["0"].keys()
    for model in [ds, gpt, qwen]:
        vote_results = {}
        for comment in comments:
            shot_0_vote = model["0"][comment]
            shot_2_vote = model["2"][comment]
            shot_4_vote = model["4"][comment]
            shot_6_vote = model["6"][comment]
            shot_10_vote = model["10"][comment]
            vote_results[comment] = vote([shot_0_vote, shot_2_vote, shot_4_vote,
                                          shot_6_vote, shot_10_vote])
        print(calculate_metrics(vote_results))

def scenario_2():
    comments = ds["0"].keys()
    for shot in ["0", "2", "4", "6", "10"]:
        vote_results = {}
        for comment in comments:
            ds_vote = ds[shot][comment]
            gpt_vote = gpt[shot][comment]
            qwen_vote = qwen[shot][comment]
            vote_results[comment] = vote([ds_vote, gpt_vote, qwen_vote])
        print(calculate_metrics(vote_results))


comments = ds["0"].keys()
def compare_for_shot():
    for shot in ["0", "2", "4", "6", "10"]:
        ds_shot_results = ds[shot]
        gpt_shot_results = gpt[shot]
        qwen_shot_results = qwen[shot]
        ds_f1, ds_r, ds_p, ds_acc = calculate_metrics(ds_shot_results)
        gpt_f1, gpt_r, gpt_p, gpt_acc = calculate_metrics(gpt_shot_results)
        qwen_f1, qwen_r, qwen_p, qwen_acc = calculate_metrics(qwen_shot_results)
        shot_best_f1 = max(ds_f1, gpt_f1, qwen_f1)
        shot_best_r = max(ds_r, gpt_r, qwen_r)
        shot_best_p = max(ds_p, gpt_p, qwen_p)
        shot_best_acc = max(ds_acc, gpt_acc, qwen_acc)
        def avg(numbers):
            return sum(numbers) / len(numbers)
        shot_avg_f1 = avg([ds_f1, gpt_f1, qwen_f1])
        shot_avg_r = avg([ds_r, gpt_r, qwen_r])
        shot_avg_p = avg([ds_p, gpt_p, qwen_p])
        shot_avg_acc = avg([ds_acc, gpt_acc, qwen_acc])
        vote_results = {}
        for comment in comments:
            ds_vote = ds[shot][comment]
            gpt_vote = gpt[shot][comment]
            qwen_vote = qwen[shot][comment]
            vote_results[comment] = vote([ds_vote, gpt_vote, qwen_vote])
        ensemble_f1, ensemble_r, ensemble_p, ensemble_acc = calculate_metrics(vote_results)
        print(f"{shot}-shot:\n")
        for mectric in ["p", "r", "f1", "acc"]:
            print("",end="& ")
            mean = eval(f"shot_avg_{mectric}")
            best = eval(f"shot_best_{mectric}")
            print('%.3f'%mean, end=" & ")
            print('%.3f'%best, end=" & ")
            ensemble_mectric = eval(f"ensemble_{mectric}")
            improve_mean = (ensemble_mectric - mean) / mean
            improve_best = (ensemble_mectric - best) / best
            print('%.3f'%ensemble_mectric, end=" & ")
            print('{:.2%}'.format(improve_mean), end=" & ")
            print('{:.2%}'.format(improve_best), end=" & ")
        print()


compare_for_shot()

def compare_for_model():
    for model in ["ds", "gpt", "qwen"]:
        m = eval(model)
        model_0_results = m["0"]
        f1_0, r_0, p_0, acc_0 = calculate_metrics(model_0_results)
        model_2_results = m["2"]
        f1_2, r_2, p_2, acc_2 = calculate_metrics(model_2_results)
        model_4_results = m["4"]
        f1_4, r_4, p_4, acc_4 = calculate_metrics(model_4_results)
        model_6_results = m["6"]
        f1_6, r_6, p_6, acc_6 = calculate_metrics(model_6_results)
        model_10_results = m["10"]
        f1_10, r_10, p_10, acc_10 = calculate_metrics(model_10_results)
        model_best_f1 = max(f1_0, f1_2, f1_4, f1_6, f1_10)
        model_best_r = max(r_0, r_2, r_4, r_6, r_10)
        model_best_p = max(p_0, p_2, p_4, p_6, p_10)
        model_best_acc = max(acc_0, acc_2, acc_4, acc_6, acc_10)
        def avg(numbers):
            return sum(numbers) / len(numbers)
        model_avg_f1 = avg([f1_0, f1_2, f1_4, f1_6, f1_10])
        model_avg_r = avg([r_0, r_2, r_4, r_6, r_10])
        model_avg_p = avg([p_0, p_2, p_4, p_6, p_10])
        model_avg_acc = avg([acc_0, acc_2, acc_4, acc_6, acc_10])
        vote_results = {}
        for comment in comments:
            shot_0_vote = m["0"][comment]
            shot_2_vote = m["2"][comment]
            shot_4_vote = m["4"][comment]
            shot_6_vote = m["6"][comment]
            shot_10_vote = m["10"][comment]
            vote_results[comment] = vote([shot_0_vote, shot_2_vote, shot_4_vote,
                                          shot_6_vote, shot_10_vote])
        ensemble_f1, ensemble_r, ensemble_p, ensemble_acc = calculate_metrics(vote_results)
        print(f"{model}:\n")
        for mectric in ["p", "r", "f1", "acc"]:
            print("",end="& ")
            mean = eval(f"model_avg_{mectric}")
            best = eval(f"model_best_{mectric}")
            print('%.3f'%mean, end=" & ")
            print('%.3f'%best, end=" & ")
            # 增量 / 原总量
            ensemble_mectric = eval(f"ensemble_{mectric}")
            improve_mean = (ensemble_mectric - mean) / mean
            improve_best = (ensemble_mectric - best) / best
            print('%.3f'%ensemble_mectric, end=" & ")
            print('{:.2%}'.format(improve_mean), end=" & ")
            print('{:.2%}'.format(improve_best), end=" & ")
        print()



        
compare_for_model()

def stastics():
    stas = {"qwen":{}, "gpt":{}, "ds":{}}
    for model in ["qwen", "gpt", "ds"]:
        m = eval(model)
        for shot in shots:
            f1, r, p, acc = calculate_metrics(m[shot])
            stas[model][shot] = (p, r, f1, acc)
    
    for model in ["qwen", "gpt", "ds"]:
        print(model)
        for metrics in [0, 1, 2, 3]:    # Pecision Recall F1-score Accuracy
            print('%.4f'%stas[model]["0"][metrics], end=" & ")
            print('%.4f'%stas[model]["2"][metrics], end=" & ")
            print('%.4f'%stas[model]["4"][metrics], end=" & ")
            print('%.4f'%stas[model]["6"][metrics], end=" & ")
            print('%.4f'%stas[model]["10"][metrics], end=" & ")

            print('%.4f'%(sum([stas[model]["0"][metrics], stas[model]["2"][metrics],
            stas[model]["4"][metrics],stas[model]["6"][metrics],
            stas[model]["10"][metrics]])/5))
        print()

stastics()