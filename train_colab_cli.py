"""Fine-tune a LoRA adapter on a Colab GPU and publish it if eval loss improves.

Run with the Google Colab CLI:
    colab run --gpu T4 --timeout 0 train_colab_cli.py

Set HF_USER below. Set HF_TOKEN locally and forward it with the Colab CLI's
--env option; the remote kernel cannot prompt for interactive input.
Training dependencies are installed in the remote runtime.
"""

import glob
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def install_dependencies():
    print("Installing training dependencies on the Colab runtime...", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "transformers>=4.46,<5",
            "peft>=0.13",
            "datasets>=3.0",
            "accelerate>=1.0",
            "huggingface_hub",
        ]
    )
    try:
        torchao_version = version("torchao")
    except PackageNotFoundError:
        return

    from packaging.version import Version

    if Version(torchao_version) < Version("0.16.0"):
        print(
            f"Removing incompatible optional torchao {torchao_version}; "
            "this script uses standard LoRA.",
            flush=True,
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]
        )


def main():
    install_dependencies()

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download, login
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU is available. Start this script with "
            "`colab run --gpu T4 --timeout 0 train_colab_cli.py`."
        )
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    hf_user = "jeet390"  # Change this to your Hugging Face username.
    model_repo = f"{hf_user}/chat-model-lora"
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name = "trl-lib/Capybara"
    n_train, n_eval, max_len = 4000, 300, 512
    out_dir, adapter_dir = "out", "adapter"

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing in the Colab runtime. Set it locally and "
            "forward it with `colab run --env HF_TOKEN=<token> ...`."
        )
    login(token=token)
    api = HfApi()
    api.create_repo(model_repo, private=True, exist_ok=True)
    api.create_branch(model_repo, branch="checkpoints", exist_ok=True)
    print(f"Repo ready: {model_repo}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_dataset(dataset_name, split=f"train[:{n_train}]")
    eval_ds = load_dataset(dataset_name, split=f"test[:{n_eval}]")

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(text, truncation=True, max_length=max_len)

    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)
    print(f"{len(train_ds)} train / {len(eval_ds)} eval examples")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float32
    ).to("cuda")
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class CheckpointToHub(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            try:
                api.upload_folder(
                    folder_path=str(checkpoint),
                    repo_id=model_repo,
                    revision="checkpoints",
                    path_in_repo="latest",
                    commit_message=f"checkpoint step {state.global_step}",
                )
            except (HfHubHTTPError, OSError) as error:
                print(f"Checkpoint upload failed; training continues: {error}")

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[CheckpointToHub()],
    )

    resume = bool(glob.glob(f"{out_dir}/checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume)

    new_loss = trainer.evaluate()["eval_loss"]
    try:
        previous_metrics = json.loads(
            Path(hf_hub_download(model_repo, "metrics.json")).read_text(
                encoding="utf-8"
            )
        )
        previous_loss = previous_metrics["eval_loss"]
    except EntryNotFoundError:
        previous_loss = float("inf")

    print(
        f"new eval_loss={new_loss:.4f} | "
        f"published eval_loss={previous_loss:.4f}"
    )

    if new_loss < previous_loss:
        version = f"v1.{len(api.list_repo_refs(model_repo).tags) + 1}"
        adapter_path = Path(adapter_dir)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        (adapter_path / "metrics.json").write_text(
            json.dumps(
                {
                    "eval_loss": new_loss,
                    "version": version,
                    "base_model": base_model,
                }
            ),
            encoding="utf-8",
        )
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=model_repo,
            commit_message=f"{version}: eval_loss {new_loss:.4f}",
        )
        api.create_tag(model_repo, tag=version)
        print(f"Pushed {version} to https://huggingface.co/{model_repo}")
    else:
        print("No improvement - nothing pushed.")

    model.eval()
    messages = [{"role": "user", "content": "Give me two tips for learning Python."}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=120, do_sample=False
        )
    reply = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(reply)


if __name__ == "__main__":
    main()
