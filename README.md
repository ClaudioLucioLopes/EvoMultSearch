# EvoMultSearch: Evolutionary Multi-objective Multi-provider Prompt-tuning

##Introduction

 The concurrent optimization of language models and instructional prompts presents a significant challenge for deploying efficient and effective AI systems, particularly when balancing performance against computational costs like token usage. This paper introduces and assesses a bi-objective evolutionary search engine designed to navigate this complex space, focusing specifically on Small Language Models (SLMs). We employ the NSGA-II algorithm and prompt grammar to simultaneously optimize for task accuracy and token efficiency across some reasoning tasks. Our results successfully identify diverse, high-performing model-prompt combinations, quantitatively revealing the critical trade-off between the two objectives. This research highlights task-specific affinities between particular SLMs and prompt structures (e.g., instructions, context, chain of thought). The generated practical Pareto fronts offer decision-makers a portfolio of optimized solutions adaptable to their specific constraints. This automated approach moves beyond traditional manual tuning, providing a foundational framework for discovering effective human-AI interaction patterns.

EvoMultSearch is a Python-based framework that uses evolutionary algorithms to optimize prompts for small language models (SLMs). The primary goal is to automatically find prompts that are both effective (maximizing accuracy) and efficient (minimizing token count). This is framed as a multi-objective optimization problem.

## Features

*   **Multi-objective Optimization:** Simultaneously optimizes for prompt accuracy and token count.
*   **Evolutionary Algorithm:** Uses the NSGA-II algorithm to evolve a population of prompts and models.
*   **Flexible Prompt Structure:** Defines prompt structures using BNF grammars.
*   **Extensible:** Easy to add new genetic operators, evaluation metrics, and LLM providers.
*   **Experiment Management:** Provides a command-line interface for running and managing experiments.

## How It Works

EvoMultSearch treats prompt engineering and model choice as a search problem. A "prompta and model" are considered an individual in a population, and it's composed of different parts (e.g., prompt: context, instructions, examples; and model). The evolutionary algorithm then iteratively applies genetic operators (crossover and mutation) to create new, hopefully better, solutions.

The optimization process is guided by two main objectives:

1.  **Accuracy:** How well the prompt performs on a given task.
2.  **Token Count:** The number of tokens in the prompt.

The framework uses the `pymoo` library for the underlying evolutionary algorithm and provides a set of classes and scripts for defining, evaluating, and evolving prompts.

## Getting Started

### Prerequisites

*   Python 3.8+
*   The required Python packages are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ClaudioLucioLopes/EvoMultSearch.git
    cd EvoMultSearch
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point for running experiments is the `PromptExperiment.py` script. You can run an experiment from the command line, specifying the BNF grammar file, the dataset, and other parameters.

```bash
python PromptExperiment.py \
    --bnf_filepath grammars/your_grammar.bnf \
    --dataset testsets/your_dataset.json \
    --runs 5 \
    --pop_size 30 \
    --gens 10
```

This will run the optimization process and save the results (the Pareto front of non-dominated solutions) to the output directory.

## Project Structure

*   `PromptExperiment.py`: The main script for running experiments.
*   `PromptIndividual.py`: Defines the structure of a single prompt.
*   `PromptCrossover.py`, `PromptMutator.py`, `PromptSampling.py`: Implement the genetic operators.
*   `PromptOptimizationProblem.py`: Defines the optimization problem and objectives.
*   `prompt_eval.py`: Handles the evaluation of prompts.
*   `grammars/`: Contains the BNF grammar files that define the prompt structures.
*   `testsets/`: Contains the datasets for evaluating the prompts.
*   `AnalyzeParetoResult.py`: A script for analyzing the results of an experiment.
*   `plot_solution.py`: A script for plotting the Pareto front.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## Citing This Work
You can cite this code as follows:


### Bibtex


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

