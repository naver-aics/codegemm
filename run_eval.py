import os
import json
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from codegemm.inference.codegemm_causallm import CodeGEMMForCausalLM
from codegemm.evaluate import eval

print("""This script will evaluate all models in the cache directory by:
    1. Calculating perplexity on specified datasets, and
    2. Evaluating downstream tasks using lm_eval on specified tasks.
    
To view and modify the datasets and tasks to be evaluated, please modify this script directly.
Also check the provided command line arguments for more options.
""")


def get_tokenizer_type(model_path):
    if 'llama-2' in model_path.lower():
        tokenizer_type = 'llama-2'
    elif 'llama' in model_path.lower():
        tokenizer_type = 'llama'
    elif 'opt' in model_path.lower():
        tokenizer_type = 'opt'
    elif 'mistral' in model_path.lower():
        tokenizer_type = 'mistral'
    elif 'phi-2' in model_path.lower():
        tokenizer_type = 'phi-2'
    elif 'gemma' in model_path.lower():
        tokenizer_type = 'gemma'
    else:
        tokenizer_type = None

    return tokenizer_type

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--output_file', type=str, default='results.json')
parser.add_argument('--redo', action='store_true')
parser.add_argument('--cache_dir', type=str, default='./cache')
parser.add_argument('--downstream', action='store_true')
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()

model_paths = []
# num_fewshot = 5


# Uncomment the line below to run baseline models
# model_paths += utils.get_base_models(include_prequant=False, relevant_models_only=True)

# model_paths += utils.get_subdirs(f'{args.cache_dir}/fake_packed')
# model_paths += utils.get_subdirs(f'{args.cache_dir}/packed')

# model_paths += args.model_path

# testcases for perplexity calculation
# datasets = ['wikitext2', 'c4_new', 'ptb_new_sliced']
datasets = ['wikitext2', 'c4_new']

# tasks for lm_eval
if args.downstream:
    tasks = ['winogrande', 'piqa', 'arc_easy', 'arc_challenge', 'hellaswag']
    # tasks = ['hellaswag']
    # tasks = []
    num_fewshot = 0
    # tasks = ['gsm8k_cot']
    # num_fewshot = 8
else:
    tasks = ['mmlu']
    num_fewshot = 5

# read previous results
if os.path.exists(args.output_file):
    with open(args.output_file) as f:
        all_results = json.load(f)
else:
    all_results = {}

new_results = {}  # results that are newly calculated, to be printed at the end

total_tests_to_run = {}  # tasks to be run will be stored here
skipped_models = []  # models that are skipped will be stored here

# Check which models/testcases need to be run
# This is done first so that we know how many tasks there are in total,
# and thus we can print the progress
# for model_path in model_paths:
for _ in range(1):
    # model_name = os.path.basename(model_path)
    model_path = args.model_path
    model_jobs = {'to_print': [], 'ppl': [], 'lm-eval': []}
    model_jobs['ppl'] = datasets
    model_jobs['lm-eval'] = tasks
    model_jobs['to_print'].append(f"Running datasets: {model_jobs['ppl']}")
    model_jobs['to_print'].append(f"Running tasks: {model_jobs['lm-eval']}")
    total_tests_to_run[args.model_path] = model_jobs

total_ppl_job_count = sum(len(model_tasks['ppl']) for model_tasks in total_tests_to_run.values())
total_lm_eval_job_count = sum(len(model_tasks['lm-eval']) for model_tasks in total_tests_to_run.values())
if skipped_models:
    print(f">> {len(skipped_models)} models will be skipped because all dataset results already exist.")
    # print('\n'.join(skipped_models) + '\n')
print(f">> Summary: {total_ppl_job_count} ppl jobs and {total_lm_eval_job_count} lm-eval tasks")
print(args.model_path)


def save_results(results_dict):
    def recursive_sort_dict(d):
        if isinstance(d, dict):
            return {k: recursive_sort_dict(v) for k, v in sorted(d.items())}
        return d

    sorted_results = recursive_sort_dict(results_dict)

    with open(args.output_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)


# Run all tasks
for i, model_path in enumerate(total_tests_to_run):
    # model_name = os.path.basename(model_path)
    model_name = model_path
    model_jobs = total_tests_to_run[model_path]
    to_print = model_jobs['to_print']
    datasets_to_evaluate = model_jobs['ppl']
    tasks_to_evaluate = model_jobs['lm-eval']
    print("==================================================")
    print(f" Model: {model_name}")
    print(f"Progress: {i + 1}/{len(total_tests_to_run)}")
    print("==================================================")
    datasets_with_results = [testcase for testcase in datasets if testcase in all_results.get(model_name, {})]

    for line in to_print:
        print('>> ' + line)

    ppl_results = {}
    lm_eval_results = {}

    # Run evaluation
    # tokenizer_type, tokenizer, model = eval.auto_model_load(model_path, args.fp16)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    # model = AnyBCQForCausalLM.from_quantized(model_path).to(device)
    model = CodeGEMMForCausalLM.from_quantized(model_path).to("cuda")
    tokenizer_type = get_tokenizer_type(model_path)

    
    
    if datasets_to_evaluate:
        ppl_results = eval.evaluate_ppl(model, tokenizer, datasets_to_evaluate, verbose=True,
                                        chunk_size=2048, tokenizer_type=tokenizer_type)

    # Update ppl results
    new_results[model_name] = {}
    if ppl_results:
        new_results[model_name]['ppl'] = ppl_results
        all_results.setdefault(model_name, {}).setdefault('ppl', {}).update(ppl_results)

    save_results(all_results)

    # Run lm_eval
    if tasks_to_evaluate:
        lm_eval_results = eval.run_lm_eval(tokenizer, model, tasks_to_evaluate, num_fewshot)

    # Update lm_eval results
    if lm_eval_results:
        new_results['lm-eval'] = lm_eval_results
        all_results.setdefault('lm-eval', {}).update(lm_eval_results)

    save_results(all_results)

    print()

    del model  # clear memory

print("---------------------- All Results ----------------------")
# print new results as formatted json
print(json.dumps(new_results, indent=4))

if len(total_tests_to_run) == 0:
    exit(1)