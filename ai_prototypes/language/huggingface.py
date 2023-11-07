from typing import List, Literal, Optional
import os
import json

import torch
import datasets
from dataclasses import dataclass, field, asdict
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    pytorch_utils,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
from peft import (
    PeftModel,
    LoraConfig,
    IA3Config,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
import bitsandbytes as bnb
from torchmetrics.text.rouge import ROUGEScore


def get_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    model_max_length: int=2048,
    fp16: bool=False,
    bf16: bool=False,
    max_memory: int=75000,
    gradient_checkpointing: bool=False,
    adapter_path: Optional[str]=None,
    peft_type: str="none",
    lora_r: int=8,
    lora_alpha: int=16,
    lora_dropout: float=0.05,
    toolken_functions_fn: str="",
    trust_remote_code: bool=False,
    training: bool=True,
    deepspeed: bool=False,
):
    # FIXME: defualt, all
    target_class_type = "default"

    if peft_type == "qlora":
        bits = 4
        target_class_type = "all"
    else:
        bits = 16 if fp16 or bf16 else 32

    # If we are in a distributed setting, we need to set the device map and max memory per device
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if training:
        device_map = None if deepspeed else {'': local_rank}
        # max_memory = {'': max_memory[local_rank]}
    else:
        if deepspeed:
            raise NotImplementedError()
        device_map = "auto"

    torch_dtype = get_torch_dtype(fp16, bf16)
    compute_dtype = get_compute_dtype(fp16, bf16)

    if peft_type == "qlora":
        load_in_4bit = bits == 4
        load_in_8bit = bits == 8
        quant_type = "nf4"
        use_double_quant = True

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            # max_memory=max_memory,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_quant_type=quant_type,
            ),
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            # max_memory=max_memory,
        )

    model.config.torch_dtype = torch_dtype

    if training and peft_type != "none":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing)

    if peft_type in ("lora", "qlora", "ia3"):
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model, adapter_path, is_trainable=training)
        else:
            target_modules = None
            feedforward_modules = None

            if target_class_type == "all":
                # FIXME:
                # Find all linear names
                target_classes = [
                    torch.nn.Linear,
                    torch.nn.Conv1d,
                    pytorch_utils.Conv1D,
                ]
                if peft_type == "qlora":
                    target_classes += [bnb.nn.Linear4bit, bnb.nn.Linear8bitLt]

                lora_module_names = set()
                for name, module in model.named_modules():
                    if any([isinstance(module, target_class) for target_class in target_classes]):
                        names = name.split('.')
                        lora_module_names.add(names[0] if len(names) == 1 else names[-1])

                if 'lm_head' in lora_module_names: # needed for 16-bit
                    lora_module_names.remove('lm_head')

                target_modules = list(lora_module_names)

            if peft_type in ("lora", "qlora"):
                peft_config = LoraConfig(
                    task_type="CAUSAL_LM",
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    target_modules=target_modules,
                )

            elif peft_type in ("ia3",):
                peft_config = IA3Config(
                    peft_type="IA3",
                    task_type="CAUSAL_LM",
                    inference_mode=not training,
                    target_modules=target_modules,
                    feedforward_modules=feedforward_modules,
                )

            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)

    # Set tensor type precision
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name and training:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    if gradient_checkpointing:
        model.config.use_cache = False

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=model_max_length,
        trust_remote_code=trust_remote_code
    )

    if tokenizer.pad_token is None:
        # FIXME: unk. we want this to be different from the eos token. especially LLaMA
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_torch_dtype(fp16, bf16):
    if fp16:
        return torch.float16
    elif bf16:
        return torch.bfloat16
    return torch.float32


def get_compute_dtype(fp16, bf16):
    if fp16:
        return torch.float16
    elif bf16:
        return torch.bfloat16
    return torch.float32
