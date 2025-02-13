import itertools
import numpy as np
from functools import partial
from models.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, question, to_print=False):
    global gpt
    gpt = partial(gpt, model="gpt-4o-mini", temperature=1.0)
    print(gpt)
    x = question  # input (# Retrieves the 4 numbers for the puzzle in Game of 24)
    ys = ['']  # current output candidates (# Empty thought to start with)

    reasoning_log = []  # List to capture the entire reasoning process
    reasoning_log.append(f"Starting reasoning process for question: {x}\n")
    
    for step in range(task.steps):
        # generation
        # The method used to generate thoughts depends on the configuration (args.method_generate).
        reasoning_log.append(f"\nStep {step + 1}: Generation\n")
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
            reasoning_log.append(f"Generated thoughts: {new_ys}\n")
            """
            the method generates new samples based on the task’s prompt format (e.g., standard or chain-of-thought (CoT) prompts). 
            Here, args.n_generate_sample controls how many thoughts are generated (in your case, 5 samples).
            """
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
            reasoning_log.append(f"Proposed thoughts: {new_ys}\n")
            """
            This method generates proposals (potential solutions) based on the current partial solution. 
            It wraps the "current numbers" and "previous steps" into a "prompt" that is passed to the GPT model, 
            which returns the next possible steps.
            """
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        
        # evaluation (The n_evaluate_sample parameter controls how many evaluations are performed per thought.)
        reasoning_log.append(f"Step {step + 1}: Evaluation\n")
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
            reasoning_log.append(f"Votes for each thought: {values}\n")
            """
            This method aggregates votes from multiple evaluations to rank each thought.
            """
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
            reasoning_log.append(f"Values for each thought: {values}\n")
            """
            The candidate solutions (new_ys) are evaluated by asking GPT to estimate the likelihood that 
            each thought can lead to the correct solution. The evaluation results are cached to avoid redundant processing.
            """

        # selection (After evaluating the thoughts, the best candidates are selected for the next iteration:)
        reasoning_log.append(f"Step {step + 1}: Selection\n")
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        reasoning_log.append(f"Selected thoughts for the next step: {select_new_ys}\n")

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    # ys: A list of selected candidate solutions (thoughts) that represent the final steps of the solution.
    # infos: Metadata about the steps, inputs, generated thoughts, evaluated values, and selected candidates.
    reasoning_log.append(f"\nFinal selected thought: {ys}\n")
    return ys, "\n".join(reasoning_log)