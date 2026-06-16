import openai
import json
import pandas as pd
from string import Template
import random
from sklearn.model_selection import train_test_split
import os

# import prompt template for llms from the `prompt_templates` folder
# if multiple prompts are imported, they will run the experiment and be stored separately
# Note that the zero-shot case needs different prompt template, see the readme file
templates = ["template_1"]
# set the shots
shots = [0,2,4,6,10]
# set the models you want to run
models = ["gpt-4o"]
# set the random seed here
RANDOM_SEED = 42
if not os.path.exists(f"./out/{str(RANDOM_SEED)}"):
    os.mkdir(f"./out/{str(RANDOM_SEED)}")

# set your LLM client here.
client = openai.OpenAI(
            api_key="[OpenAI API KEY]",
            base_url="[URL]",
        )
    

# get response from the chosen LLM
def get_resp(prompt, model):
    resp = None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
            # some LLMs do not support the parameters like `top_p` and `n`
            # so that they may not respond the same each time you call it
            max_tokens=50,
            temperature=0.0, 
            top_p=1.0,
            n=1,
            stop=None
        )
    except:
        print(f"ERROT!!!!!\nPROMPT:{prompt}")
        return "wrong response."
    
    return resp.choices[0].message.content


# we assume that the prompt requires LLM to give its answer in a pair of backticks
# such as `violation` or `non-violation`
# so this method will retry if the format of the response is inappropriate
def get_usable_resp(prompt, model):
    count = 0
    resp = ""
    while resp!="`violation`" and resp!="`non-violation`" and count<10:
        resp = get_resp(prompt, model)
        # the `deepseek-reasoner` model will return differently, which means it will return with a excessive blank space at the very beginning
        if model=="deepseek-reasoner":
            resp = resp[2:]
        count += 1
        # If LLM retried more than 10 times, we regard it as failing to resolve the prompt given 
    if count>= 10:
        return "fail."

    if resp == "`violation`":
        return True
    else:
        return False
    
# read data and split train set and test set
with open("./dataset/Violation_symptoms.xlsx", mode='rb') as f:
    violation_frame = pd.read_excel(f)

with open(".dataset/Randomly_selected_comments.xlsx", mode='rb') as f:
    non_violation_frame = pd.read_excel(f)


violations = violation_frame["Comment"].tolist()
non_violation = non_violation_frame["Comment"].tolist()

checker = {}
for violation in violations:
    checker[violation] = True
for non_vio in non_violation:
    checker[non_vio] = False

def check_violation(comment):
    return checker.get(comment, None)   


train_set_violation, test_set_violation = train_test_split(violations, test_size=0.2, train_size=0.8, random_state=RANDOM_SEED)
train_set_non_violation, test_set_non_violation = train_test_split(non_violation, test_size=0.2, train_size=0.8, random_state=RANDOM_SEED)
train_set = train_set_violation + train_set_non_violation
test_set = test_set_violation + test_set_non_violation

with open(f"./out/{str(RANDOM_SEED)}/train_set.json", mode="w", encoding='utf-8') as f:
    json.dump(train_set, f)

with open(f"./out/{str(RANDOM_SEED)}/test_set.json", mode="w", encoding='utf-8') as f:
    json.dump(test_set, f)


# shot_num should be an even number to make sure the balance
# select examples to fill the prompt template
def select_shots(shot_num):
    shots = []
    half_shot_num = int(shot_num / 2)
    violation_shots = random.sample(train_set_violation, half_shot_num)
    non_violation_shots = random.sample(train_set_non_violation, half_shot_num)
    count = 1
    for i in range(half_shot_num):
        shots.append(f"Comment {str(count)}:\n{violation_shots[i]}\nClassification: violation\n\n")
        count += 1
        shots.append(f"Comment {str(count)}:\n{non_violation_shots[i]}\nClassification: non-violation\n\n")
        count += 1
    
    return "".join(shots)


def dump_results(results, model):
    with open(f"./out/{str(RANDOM_SEED)}/results-{model}.json", mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":

    for model in models:
        results = {}
        try:
            for template_name in templates:
                with open(f"./prompt_templates/{template_name}.txt", mode="r", encoding="utf-8") as f:
                    template = Template(f.read())
                results[template_name] = {}
                for shot_num in shots:
                    if shot_num == 0:
                        with open(f"./prompt_templates/{template_name}_zero_shot.txt", mode="r", encoding="utf-8") as f:
                            template = Template(f.read())
                    shot_content = select_shots(shot_num)
                    results[template_name][shot_num] = {}
                    count = 1
                    for test_case in test_set:
                        if shot_num != 0:
                            full_prompt = template.substitute(shot_content=shot_content, test_case=test_case)
                        else:
                            full_prompt = template.substitute(test_case=test_case)
                        resp = get_usable_resp(full_prompt, model)
                        if resp == "fail.": # LLM fails to resolve
                            print(test_case)
                            print(f"{str(shot_num)}-Shot #{str(count)} : fail to fetch ans.")
                            results[template_name][shot_num][test_case] = False
                        else:   # success
                            results[template_name][shot_num][test_case] = resp
                        print(f"{str(shot_num)}-Shot #{str(count)} : {str(resp)}")
                        count += 1
        finally:
            dump_results(results, model)
        
    
