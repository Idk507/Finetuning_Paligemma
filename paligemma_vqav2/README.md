---
license: gemma
library_name: peft
tags:
- generated_from_trainer
base_model: leo009/paligemma-3b-pt-224
datasets:
- vq_av2
model-index:
- name: paligemma_vqav2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# paligemma_vqav2

This model is a fine-tuned version of [leo009/paligemma-3b-pt-224](https://huggingface.co/leo009/paligemma-3b-pt-224) on the vq_av2 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 2
- num_epochs: 2

### Training results



### Framework versions

- PEFT 0.10.0
- Transformers 4.42.0.dev0
- Pytorch 2.2.2+cu118
- Datasets 2.18.0
- Tokenizers 0.19.1