import os
import argparse
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="/sas_data/project/LLMP/model/llama-hf/llama-7b-hf")
    parser.add_argument("-t", "--tokenizer", type=str, default="/sas_data/project/LLMP/model/llama-hf/llama-tokenizer")
    parser.add_argument("-d", "--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--dataset_config", type=str, default="3.0.0")
    parser.add_argument("-o", "--output", type=str, default="outputs/")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    lr = 1e-3
    batch_size = 1
    num_epochs = 4
    max_length = 2048
    text_column = "article"
    label_column = "highlights"

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def preprocess(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column}: {x}\n{label_column}: " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]

        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids

            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]

            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = load_dataset(args.dataset, args.dataset_config)
    tokenized_dataset = dataset.map(preprocess, batched=True)
    train_dataset = tokenized_dataset['train']
    test_dataset = tokenized_dataset['test']

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    rouge = evaluate.load('rouge')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    training_args = TrainingArguments(
        output_dir=args.output,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
