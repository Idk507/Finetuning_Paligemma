# Finetuning Paligemma

This repository contains code for fine-tuning the `paligemma-3b-pt-224` model using the Peft framework for the Visual Question Answering (VQA) task.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [References](#references)
- [License](#license)

## Overview

This project utilizes the Hugging Face Transformers library and the Peft framework to fine-tune a large pre-trained model (`paligemma-3b-pt-224`) for the VQA task. The configuration has been updated to use `gelu_pytorch_tanh` as the hidden activation function.

## Setup

### Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Peft

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Idk507/Finetuning_Paligemma.git
cd Finetuning_Paligemma
pip install -r requirements.txt
```

### Usage

Loading the Pre-trained Model

To load the pre-trained model and configuration, use the following code:
```
from transformers import AutoModelForPreTraining
from peft import PeftConfig, PeftModel

# Load the configuration and update the hidden activation
config = PeftConfig.from_pretrained("Dhanushkumar/paligemma_VQAv2")
config.hidden_activation = "gelu_pytorch_tanh"  # Set the desired activation function

# Load the base model
base_model = AutoModelForPreTraining.from_pretrained("leo009/paligemma-3b-pt-224", config=config)

# Wrap the base model with PeftModel
model = PeftModel.from_pretrained(base_model, "Dhanushkumar/paligemma_VQAv2")

```


