# num_seq_env

Single-turn inductive reasoning environment over numeric sequences governed by variable-order linear recurrence relations. Built with Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

## Setup

```bash
prime env install num-seq-env
```

## Usage

```bash
# Install locally
prime env install num-seq-env

# Run evaluation
prime eval run num-seq-env

# Push to Prime Hub
prime env push -p ./environments/num_seq_env
```

## Environment

| Environment | Description |
| ----------- | ----------- |
| [num-seq-env](environments/num_seq_env/) | Single-turn environment where the model is shown consecutive terms from a linear recurrence sequence and must compute a specific term by its absolute position. The sequence order k is sampled from {2, 3, 4, 5} with randomized coefficients and initial values. Reward is exact match. |
