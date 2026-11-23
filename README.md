# HT-MIA

Official implementation of **“What Hard Tokens Reveal: Exploiting Low-confidence Tokens for Membership Inference Attacks against Large Language Models,”** accepted to Findings of EACL 2027.

- Paper: [arXiv:2601.20885](https://arxiv.org/abs/2601.20885)
- Method: HT-MIA compares token-level probabilities from a fine-tuned target model and its pretrained reference model, focusing on the target model’s lowest-confidence tokens.

## Repository structure

```text
attacks/                         HT-MIA and baseline attack implementations
finetune/                        Data preparation and model fine-tuning scripts
config.yaml                      Attack and evaluation configuration
create_membership_datasets.py    Member/non-member evaluation-set creation
main.py                          Attack evaluation entry point
requirements.txt                 Python dependencies
```

## Installation

Python 3.10 or later and a CUDA-capable GPU are recommended. Create an isolated environment and install the dependencies:

```bash
git clone https://github.com/006jawad/ht-mia-source.git
cd ht-mia-source

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some models, including Llama models, require accepting their access conditions on Hugging Face. Authenticate without placing a token in this repository:

```bash
hf auth login
```

## Reproducing the workflow

### 1. Prepare the datasets

The paper uses a 50:50 member/non-member split. Prepare either of the medical datasets as follows:

```bash
python finetune/medical_data_prep.py \
  --dataset asclepius \
  --format simple \
  --train_ratio 0.5 \
  --output_dir ./medical_processed
```

Use `clinicalnotes` instead of `asclepius` for the other medical dataset. For the general-domain datasets, use:

```bash
python finetune/general_data_prep.py \
  --dataset imdb \
  --format simple \
  --train_ratio 0.5 \
  --output_dir ./general_processed
```

Use `wikipedia` instead of `imdb` for the other general-domain dataset. The scripts download the public datasets through Hugging Face and save `train_simple` and `val_simple` datasets locally.

### 2. Fine-tune a target model

The following example fine-tunes Qwen3-0.6B using the paper’s main training settings:

```bash
torchrun --nproc_per_node=1 --master_port=29500 finetune/finetune_qwen.py \
  --model_name Qwen/Qwen3-0.6B-Base \
  --data_path ./medical_processed/asclepius \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --max_length 512 \
  --output_dir ./models/qwen3-0.6b-asclepius
```

Related scripts are provided for GPT-2 (`finetune_gpt.py`), Llama-3.2-1B (`finetune_llama.py`), and the Llama-3.1-8B/70B scaling experiments (`finetune_llama_scaling.py`). Adjust the number of processes and batch size to match the available GPUs.

### 3. Create the MIA evaluation dataset

Create the balanced member/non-member dataset from the same processed data:

```bash
python create_membership_datasets.py \
  --data_dir ./medical_processed \
  --datasets asclepius \
  --format simple \
  --output_dir ./membership_data
```

This produces `./membership_data/asclepius_simple_mia`. For IMDB or Wikipedia, change `--data_dir` to `./general_processed` and select the corresponding dataset name.

### 4. Configure and run HT-MIA

In `config.yaml`, set:

- `global.target_model` to the fine-tuned model directory.
- `global.datasets[0].name` to the MIA dataset directory, including the leading `./` for a local path.
- `global.device` to the target-model device.
- `ht_mia.base_model_name` to the corresponding pretrained reference model.
- `ht_mia.base_device` to the reference-model device.

Then run:

```bash
mkdir -p results
python main.py \
  --config config.yaml \
  --attacks ht_mia \
  --seed 42 \
  --output ./results/ht_mia.pkl
```

The program reports AUC, TPR@FPR=0.1, and TPR@FPR=0.01, including bootstrap mean and standard deviation.

## Baselines and additional comparisons

Run every attack configured in `config.yaml` with:

```bash
python main.py \
  --config config.yaml \
  --run-all \
  --seed 42 \
  --output ./results/all_attacks.pkl
```

The repository includes Loss, Zlib, Lowercase, Min-K++, Ratio, PAC, EZ-MIA, and WBC implementations. The SPV-MIA implementation used for comparison is maintained in its [official repository](https://github.com/tsinghua-fib-lab/ANeurIPS2024_SPV-MIA).

## Differentially private training

The repository includes `finetune_gpt2_DP.py` and `finetune_qwen_DP.py`. For example, the paper’s main defense setting for Qwen uses:

```bash
python finetune/finetune_qwen_DP.py \
  --model_name Qwen/Qwen3-0.6B-Base \
  --data_path ./medical_processed/asclepius \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --max_length 512 \
  --target_epsilon 100.0 \
  --target_delta 1e-5 \
  --max_grad_norm 100.0 \
  --output_dir ./models/qwen3-0.6b-asclepius-dp
```

## Citation

```bibtex
@article{jawad2026hard,
  title   = {What Hard Tokens Reveal: Exploiting Low-confidence Tokens for Membership Inference Attacks against Large Language Models},
  author  = {Jawad, Md Tasnim and Xiao, Mingyan and Wu, Yanzhao},
  journal = {arXiv preprint arXiv:2601.20885},
  year    = {2026}
}
```

The proceedings citation will replace the preprint citation after the EACL 2027 Anthology entry is available.

## License

This project is released under the [MIT License](LICENSE).

## Third-party attribution

The WBC comparison is adapted from the MIT-licensed WBC implementation. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and license terms.

