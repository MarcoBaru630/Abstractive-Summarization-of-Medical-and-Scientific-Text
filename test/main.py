from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
import evaluate
from src.settings import BASE_DIR

from peft import LoraConfig, get_peft_model, TaskType

if __name__ == "__main__":
    from src.preprocessing import Preprocessor, tokenizer, model
    import torch

    torch.set_num_threads(torch.get_num_threads())  # ensures PyTorch uses all threads
    print("Torch is using", torch.get_num_threads(), "CPU threads")
    def print_number_of_trainable_model_parameters(_model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in _model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE expects sentences to be separated by escapes characters
        decoded_preds = ['\n'.join(p.strip().split('. ')) for p in decoded_preds]
        decoded_labels = ['\n'.join(l.strip().split('. ')) for l in decoded_labels]

        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

        return {k:round(v, 2)*100 for k, v in result.items()}



    processed_data = Preprocessor()

    train_data = processed_data.train()
    valid_data = processed_data.eval()
    test_data = processed_data.test()


    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100, # only value for padding ignored during cross entropy, whereas zero are counted
    )

    lora_config = LoraConfig(
        r = 8, # maybe 16 for summarization
        lora_alpha = 8, # we don't have a lot of memory
        target_modules= ["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias = "none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    peft_model = get_peft_model(model, lora_config)

    print(print_number_of_trainable_model_parameters(peft_model))



    peft_training_args = Seq2SeqTrainingArguments(
        output_dir= BASE_DIR / "output",
        save_strategy="steps",
        save_steps=1200,
        eval_strategy="steps",
        eval_steps=300,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
        save_total_limit=2,
        load_best_model_at_end=True,
        num_train_epochs=1,
        logging_dir= BASE_DIR / "logs",
        logging_strategy="steps",
        logging_steps=1200,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        report_to="none",
        dataloader_num_workers=8,
    )


    trainer = Seq2SeqTrainer(
        args=peft_training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        model=peft_model,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    trainer.model.save_pretrained(BASE_DIR / "model")
    tokenizer.save_pretrained(BASE_DIR / "model")
