import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#task = sys.argv[1]

#tasks = ['epistemic_reasoning','implicatures','logical_fallacy_detection','navigate','snarks','winowhy','hyperbaton'] #causal_judgment
tasks = ['snarks']
#prompts = ['Determine the correlation between the stated premises and hypotheses explicitly. Please, answer with only "entailment" or "non-entailment". For instance:\nQUESTION: Premise: David suspects that a bald man is getting out of a small blue car. Hypothesis: David suspects that the man is bald.\nANSWER: entailment\n[[INPUT]]','[[INPUT]]\nShould Speaker 2 encapsulate their answer in a yes or no? Please, answer with only "yes" or "no". For instance:\nQUESTION: Speaker 1: \'You really hearing voices?\' Speaker 2: \'Just one.\'\nANSWER: yes\nWe could analyze this in a systematic and organized manner.','Is the logical integrity of this statement up for verification? Please, answer with only "yes" or "no". For instance:\nQUESTION: No khavvins are novalies. Some novalies are zapsters. Therefore some zapsters are khavvins.\nANSWER: no\nQUESTION: Do you think the following argument is \'Valid\' or \'Invalid\'? No sinpuds are younjurs. Some fluffsters are younjurs. Therefore no fluffsters are sinpuds.\nANSWER: no\n[[INPUT]]','If you follow these instructions, do you return to the starting point eventually? Please, answer with only "yes" or "no".\n[[INPUT]] For instance:\nQUESTION: Always face forward. Take 3 steps backward. Take 5 steps backward. Take 7 steps forward. Take 1 step backward. Take 7 steps backward. Take 6 steps forward. Take 1 step forward. Take 9 steps forward. Take 7 steps backward.\nANSWER: yes\nQUESTION: Always face forward. Take 8 steps backward. Take 9 steps right. Take 2 steps backward. Take 7 steps forward. Take 4 steps forward. Take 6 steps backward. Take 5 steps backward. Take 5 steps right. Take 6 steps left.\nANSWER: no\nWe should tackle this by taking each step with thorough and thoughtful consideration.','Unearth the statement that carries a sarcastic implication. Please, answer with only "(a)" or "(b)". For instance:\nQUESTION: (a) If I associate something bad witih it, then my depression will never let me forget it. Weirdest memorization plan ever. (b) If I associate something bad witih it, then my depression will never let me forget it. Best memorization plan ever.\nANSWER: (b)\n[[INPUT]]','Consider these examples:\nQUESTION: Bob paid for Charlie\'s college education, but now Charlie acts as though it never happened. He is very ungrateful. The \'He\' refers to charlie because of the way he has treated Charlie since the beginning.\nANSWER: no\nPlease address the inquiries about the words to which certain pronouns refer in the provided text. Please, answer with only \"yes\" or \"no\".\nConsidering the various factors and variables, let\'s think about this step by step.\n[[INPUT]]','Ascertain which sentence has the right order for its adjectives. Please, answer with only "a" or "b". For instance:\nQUESTION: a " white terrible square brand-new driving chair " b " terrible brand-new square white driving chair " ?\nANSWER: b\nConsidering the various factors and variables, let\'s think about this step by step.[[INPUT]]']
prompts = ['Which sentence has the correct adjective order? Answer with only "(a)" or "(b)". Let\'s think step by step.\n[[INPUT]]']#,'Which sentence has the correct adjective order? Answer with only "a" or "b". Let\'s think step by step.\n[[INPUT]]']


model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
#model_name_or_path = "TheBloke/Starling-LM-7B-alpha-GPTQ"
# model_name_or_path = "TheBloke/vicuna-7B-v1.5-GPTQ"

# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

for j,task in enumerate(tasks):
    individual = prompts[j]

    with open('testsets/{}.json'.format(task),'r') as f:
        tests = f.readlines()
    tests = tests[:10]
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = len(tests)
    total_input_tokens = 0
    total_output_tokens = 0
    

    for s in tests:
        js = json.loads(s)
        test = individual.replace('[[INPUT]]', 'QUESTION: ' + js['input'])

        prompt_template=f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {test} ASSISTANT:"
        #prompt_template=f"GPT4 Correct User: {test}<|end_of_turn|>GPT4 Correct Assistant:"

        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        print(input_ids,'---------------------------NUM:',len(input_ids[0]))
        total_input_tokens += len(input_ids[0])


        output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=32)#512)

        ans = tokenizer.decode(output[0])
        total_output_tokens += len(output[0])

        print(ans, file=sys.stderr)

        #answer = ans[ans.index("GPT4 Correct Assistant:")+23:-15].lower()
        
        answer = ans[ans.index("ASSISTANT:")+10:-4].lower()
        print("---------------------------------answer: ",answer)

        if task == 'causal_judgment':
            if js['target_scores']['Yes'] == 1:
                target = 'yes'
            else:
                target = 'no'

            if target == 'yes':
                if target in answer[:20]:
                    tp += 1
                else:
                    fn += 1
            else:
                if target in answer[:20]:
                    tn += 1
                else:
                    fp += 1
        elif task == 'navigate':
            if js['target_scores']['True'] == 1:
                target = 'yes'
            else:
                target = 'no'

            if target == 'yes':
                if target in answer[:20]:
                    tp += 1
                else:
                    fn += 1
            else:
                if target in answer[:20]:
                    tn += 1
                else:
                    fp += 1

        elif task == 'logical_fallacy_detection':
            if js['target_scores']['Valid'] == 1:
                target = 'yes'
            else:
                target = 'no'

            if target == 'yes':
                if target in answer[:20]:
                    tp += 1
                else:
                    fn += 1
            else:
                if target in answer[:20]:
                    tn += 1
                else:
                    fp += 1

            
        elif task == 'implicatures':
            if js['target_scores']['yes'] == 1:
                target = 'yes'
            else:
                target = 'no'

            if target == 'yes':
                if target in answer[:20]:
                    tp += 1
                else:
                    fn += 1
            else:
                if target in answer[:20]:
                    tn += 1
                else:
                    fp += 1

        if task == 'epistemic_reasoning':
            if js['target_scores']['entailment'] == 1:
                target = 'entailment'
            else:
                target = 'non-entailment'

            if target == 'non-entailment':
                if target in answer[:30]:
                    tn += 1
                else:
                    fp += 1
            else:
                if 'non' not in answer[:30]:
                    tp += 1
                else:
                    fn += 1

        if task == 'winowhy':
            if js['target_scores']['correct'] == 1:
                target = 'yes'
            else:
                target = 'no'

            if target == 'yes':
                if target in answer[:20]:
                    tp += 1
                else:
                    fn += 1
            else:
                if target in answer[:20]:
                    tn += 1
                else:
                    fp += 1

        if task == 'snarks':
            if js['target_scores']['(a)'] == 1:
                target = '(a)'
            else:
                target = '(b)'

            if target == '(a)':
                if target in answer[:20]:
                    tp += 1
                else:
                    fn += 1
            else:
                if target in answer[:20]:
                    tn += 1
                else:
                    fp += 1

        if task == 'hyperbaton':
            if js['target_scores']['a'] == 1:
                target = 'a'
            else:
                target = 'b'

            if target == 'a':
                if target in answer[-3:]:
                    tp += 1
                else:
                    fn += 1
            else:
                if target in answer[-3:]:
                    tn += 1
                else:
                    fp += 1

    accuracy = (tp+tn)/total
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*tp /(2*tp + fp + fn)

    print(task, total, accuracy, precision, recall, f1)
    print(total_input_tokens, total_output_tokens)
