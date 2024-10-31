# Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent

## Abstract

Multi-agent strategies have emerged as a promising approach to enhance the reasoning abilities of Large Language Models (LLMs) by assigning specialized roles in the problem-solving process. Concurrently, Tree of Thoughts (ToT) methods have shown potential in improving reasoning for complex question-answering tasks by exploring diverse reasoning paths.

A critical limitation in multi-agent reasoning is the 'Reasoner' agent's shallow exploration of reasoning paths. While ToT strategies could help mitigate this problem, they may generate flawed reasoning branches, which could harm the trustworthiness of the final answer.

To leverage the strengths of both multi-agent reasoning and ToT strategies, we introduce a novel approach combining ToT-based Reasoner agents with a Thought Validator agent. Multiple Reasoner agents operate in parallel, employing ToT to explore diverse reasoning paths. The Thought Validator then scrutinizes these paths, considering a Reasoner's conclusion only if its reasoning is valid.

This method enables a more robust voting strategy by discarding faulty reasoning paths, enhancing the system's ability to tackle tasks requiring systematic and trustworthy reasoning. Our method demonstrates superior performance compared to existing techniques when evaluated on the GSM8K dataset, outperforming the standard ToT strategy by an average 5.6% across four LLMs.

![Figure](https://github.com/SecureAIAutonomyLab/MA-ToT/blob/main/figures/figure.png)

## Basic Information

- **Title**: Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent
- **Authors**: Fatemeh Haji, Mazal Bethany, Maryam Tabar, Jason Chiang, Anthony Rios, Peyman Najafirad
- **Description**: This paper presents a novel multi-agent approach that combines Tree of Thoughts (ToT) with a Thought Validator agent to improve reasoning abilities in LLMs, resulting in a performance boost on the GSM8K dataset.

## Code Structure

The repository contains the following main components:

- **methods/**: Contains core methods used for generating reasoning in ToT
  - `bfs.py`: Includes functions for Thought Generation, State Evaluation and Path Selection.

- **models/**: This folder includes the model interaction components
  - `models.py`: Used for interacting with the OpenAI API and handling GPT completions

- **prompts/**: Stores the prompt templates

- **tasks/**: Implements the tasks
  - GSM8K task implementation

- **run_gsm8k_multiple_reasoners.py**: Main script to run reasoning and validation using multiple reasoners

## Installation

### Prerequisites
- Python >= 3.8
- OpenAI library version 0.27.7
- Requests version 2.31.0
- NumPy version 1.24.3
- Backoff version 2.2.1

### Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Set up the OpenAI API key:
```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

2. Set the model name in `run_gsm8k_multiple_reasoners.py`

3. Set parameters needed in bash file `run_gsm8k.sh`.

4. Run the bash file:
```bash
python run_gsm8k.sh
```

### Parameters

- `--method_generate`: Method to generate new thoughts (e.g., sample or propose)
- `--method_evaluate`: Method to evaluate thoughts (e.g., vote or value)
- `--method_select`: Method to select thoughts for the next iteration (e.g., greedy or sample)


## Data

- Dataset: GSM8K
- Format: JSONL
  - Each line represents a problem with question and answer fields
- Public data available from the GSM8K dataset source


## Citation

If you use this code in your research, please cite:

```bibtex
@article{haji2024improving,
    title={Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent},
    author={Haji, Fatemeh and Bethany, Mazal and Tabar, Maryam and Chiang, Jason and Rios, Anthony and Najafirad, Peyman},
    journal={arXiv preprint arXiv:2409.11527},
    year={2024}
}
```
