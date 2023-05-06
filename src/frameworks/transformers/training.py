from transformers import TrainingArguments, EarlyStoppingCallback, Trainer
from . import utils


def train(model, dataset_train, dataset_valid = None, batch_size: int = 32, patience: int = 3):
    training_args = TrainingArguments(
        output_dir='../models/deberta',
        evaluation_strategy='steps',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_train_epochs=3.0,
        lr_scheduler_type='linear',
        warmup_ratio=0.0,
        warmup_steps=0,
        logging_steps=50,
        save_strategy='steps',
        save_total_limit=2,
        # use_mps_device=True,  # remove for CUDA
        seed=42,
        dataloader_drop_last=False,
        run_name=None,
        disable_tqdm=None,
        remove_unused_columns=True,
        label_names=None,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=None,
        report_to='none',  # disable logging to w&b
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        compute_metrics=utils.compute_metrics,
        callbacks=callbacks,
    )
    output = trainer.train()
    metadata = output.metrics.copy()
    metadata['global_step'] = output.global_step
    return trainer
