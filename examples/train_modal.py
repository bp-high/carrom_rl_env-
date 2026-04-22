"""GRPO training for the Carrom agent on Modal — Unsloth + TRL.

Trains an instruction-tuned LLM to play Carrom under ICF rules using
Group Relative Policy Optimization (GRPO) via TRL's GRPOTrainer, with
Unsloth's 4-bit quantization and gradient-checkpointing for memory efficiency.

Cost estimates (Modal pricing, A10G @ ~$1.10/hr):
  --steps  200  (quick smoke-test)  ≈  15 min  ≈ $0.30
  --steps  500  (light training)    ≈  35 min  ≈ $0.65
  --steps 2000  (blog-quality run)  ≈  2.5 hr  ≈ $2.75
  A100 40GB @ ~$3.72/hr: --steps 2000 ≈ 1.5 hr ≈ $5.60

All budgets are well under $25.

Usage
-----
# Quickstart (200 steps, Gemma-3-4B, A10G)
modal run examples/train_modal.py

# Blog-quality run, push to HF Hub
modal run examples/train_modal.py --steps 2000 --push --repo your-username/carrom-grpo-gemma

# Larger model on A100
modal run examples/train_modal.py --model qwen2.5-3b --gpu a100 --steps 2000

# Just preview cost without running
modal run examples/train_modal.py --dry-run

Prerequisites
-------------
1. Install Modal: pip install modal && modal setup
2. Create an HF token secret in Modal dashboard:
       modal secret create hf-token HF_TOKEN=hf_...
3. (Optional) Create a W&B secret for experiment tracking:
       modal secret create wandb WANDB_API_KEY=...
"""

from __future__ import annotations

import sys
import modal

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Key → (unsloth_model_id, min_vram_gb, recommended_gpu)
MODELS: dict[str, tuple[str, int, str]] = {
    "gemma-3-1b":   ("unsloth/gemma-3-1b-it",               8,  "a10g"),
    "gemma-3-4b":   ("unsloth/gemma-3-4b-it",               16, "a10g"),  # ← default
    "qwen2.5-1.5b": ("unsloth/Qwen2.5-1.5B-Instruct",       8,  "a10g"),
    "qwen2.5-3b":   ("unsloth/Qwen2.5-3B-Instruct",         12, "a10g"),
    "qwen2.5-7b":   ("unsloth/Qwen2.5-7B-Instruct",         24, "a100"),
}

GPU_MAP = {
    "a10g": modal.gpu.A10G(),
    "a100": modal.gpu.A100(),
}

DEFAULT_MODEL = "gemma-3-4b"
HF_REPO_DEFAULT = "your-username/carrom-grpo-agent"

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------

app = modal.App("carrom-grpo")

training_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        # Core training stack
        "unsloth",
        "trl>=0.15",
        "transformers>=4.45",
        "datasets>=3.0",
        "accelerate>=0.35",
        "peft>=0.12",
        "bitsandbytes>=0.43",
        "huggingface_hub>=0.24",
        # Optional experiment tracking
        "wandb",
        # Carrom env dependencies
        "pymunk>=6.5",
        "numpy>=1.24",
        "pydantic>=2.0",
    )
)

# Mount local carrom code so the training function can import it
carrom_mount   = modal.Mount.from_local_dir(
    local_path="carrom_env",
    remote_path="/root/carrom_env",
    condition=lambda p: not any(s in p for s in ["__pycache__", ".pyc", ".pyo"]),
)
examples_mount = modal.Mount.from_local_dir(
    local_path="examples",
    remote_path="/root/examples",
    condition=lambda p: not any(s in p for s in ["__pycache__", ".pyc", ".pyo"]),
)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu=modal.gpu.A10G(),        # overridden at call time via .with_options()
    timeout=7_200,               # 2-hour cap
    secrets=[
        modal.Secret.from_name("hf-token", required=False),
        modal.Secret.from_name("wandb",    required=False),
    ],
    mounts=[carrom_mount, examples_mount],
)
def _train_remote(
    model_key:          str   = DEFAULT_MODEL,
    max_steps:          int   = 200,
    num_train_samples:  int   = 400,
    num_generations:    int   = 4,
    learning_rate:      float = 5e-6,
    push_to_hub:        bool  = False,
    hf_repo:            str   = HF_REPO_DEFAULT,
    wandb_project:      str   = "carrom-grpo",
) -> dict:
    """Core training logic executed remotely on Modal."""
    import os, random, math, json, time
    import sys
    sys.path.insert(0, "/root")

    import torch
    from datasets import Dataset
    from huggingface_hub import login
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    from carrom_env.env import CarromEnv
    from carrom_env.models import Action
    from examples.grpo_utils import (
        format_chat_prompt,
        parse_response,
        CARROM_SYSTEM_PROMPT,
    )

    # Auth
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    # W&B
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"])

    model_id, _min_vram, _rec_gpu = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Model   : {model_id}")
    print(f"Steps   : {max_steps}")
    print(f"Samples : {num_train_samples}")
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load model with Unsloth
    # ------------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------
    # 2. Generate diverse training board states
    # ------------------------------------------------------------------
    def build_dataset(n: int) -> Dataset:
        samples = []
        for seed in range(n * 3):            # oversample; filter finished games
            if len(samples) >= n:
                break
            env = CarromEnv(seed=seed)
            obs = env.reset()
            for _ in range(random.randint(0, 6)):
                a = Action(
                    placement_x=random.uniform(-0.3, 0.3),
                    angle=random.uniform(-0.9, 0.9),
                    force=random.uniform(0.25, 0.85),
                )
                obs, _, done, trunc, _ = env.step(a)
                if done or trunc:
                    break
            if obs.remaining_coins > 2:
                samples.append({"prompt": format_chat_prompt(obs)})
        print(f"Dataset: {len(samples)} board states")
        return Dataset.from_list(samples)

    train_ds = build_dataset(num_train_samples)

    # ------------------------------------------------------------------
    # 3. Reward function (ICF-aware)
    # ------------------------------------------------------------------
    def carrom_reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            text   = completion[-1]["content"] if isinstance(completion, list) else str(completion)
            reward = 0.0
            action = parse_response(text)
            if action is not None:
                reward += 0.3
                if -0.4 <= action.placement_x <= 0.4:              reward += 0.1
                if 0.15 <= action.force        <= 0.9:             reward += 0.1
                if -math.pi / 2 <= action.angle <= math.pi / 2:   reward += 0.1
                try:
                    env = CarromEnv(seed=abs(hash(text)) % 100_000)
                    env.reset()
                    _, env_reward, _, _, info = env.step(action)
                    reward += env_reward
                    reward += 0.5  * int(info.get("coin_potted", 0) > 0)
                    reward -= 0.3  * info.get("due_coins", 0)   # penalise ICF dues
                    reward -= 0.75 * info.get("foul", 0)        # penalise fouls
                except Exception:
                    pass
            else:
                reward -= 0.5
            rewards.append(reward)
        return rewards

    # ------------------------------------------------------------------
    # 4. GRPO training config
    # ------------------------------------------------------------------
    config = GRPOConfig(
        output_dir="/root/carrom-grpo-output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=num_generations,
        max_prompt_length=512,
        max_completion_length=256,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=5,
        save_steps=max(max_steps // 4, 50),
        report_to="wandb" if use_wandb else "none",
        run_name=f"carrom-grpo-{model_key}",
    )
    if use_wandb:
        config.wandb_project = wandb_project

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=carrom_reward,
        args=config,
        train_dataset=train_ds,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # ------------------------------------------------------------------
    # 6. Save / push
    # ------------------------------------------------------------------
    save_path = "/root/carrom-grpo-output/final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    result = {
        "model_id":    model_id,
        "steps":       max_steps,
        "elapsed_min": round(elapsed / 60, 1),
        "saved_to":    save_path,
    }

    if push_to_hub and hf_repo and hf_token:
        trainer.model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
        print(f"Pushed to  : https://huggingface.co/{hf_repo}")
        result["hf_repo"] = hf_repo
    else:
        print("Skipping HF push (pass --push --repo <username/repo> to enable)")

    return result


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str  = DEFAULT_MODEL,
    steps: int  = 200,
    samples: int = 400,
    generations: int = 4,
    lr: float   = 5e-6,
    gpu: str    = "a10g",
    push: bool  = False,
    repo: str   = HF_REPO_DEFAULT,
    wandb_project: str = "carrom-grpo",
    dry_run: bool = False,
):
    """
    modal run examples/train_modal.py [OPTIONS]

    --model     gemma-3-4b | gemma-3-1b | qwen2.5-1.5b | qwen2.5-3b | qwen2.5-7b
    --steps     200 (smoke) | 500 (light) | 2000 (blog quality)
    --gpu       a10g (~$1.10/hr) | a100 (~$3.72/hr)
    --push      Push trained model to HF Hub
    --repo      HF Hub repo id, e.g. myuser/carrom-grpo-gemma
    --dry-run   Print config and estimated cost, then exit
    """
    if model not in MODELS:
        print(f"Unknown model '{model}'. Choose from: {', '.join(MODELS)}")
        sys.exit(1)
    if gpu not in GPU_MAP:
        print(f"Unknown GPU '{gpu}'. Choose from: {', '.join(GPU_MAP)}")
        sys.exit(1)

    # Cost estimate (approximate)
    gpu_rate = {"a10g": 1.10, "a100": 3.72}[gpu]
    min_per_step = {"gemma-3-1b": 0.04, "gemma-3-4b": 0.08,
                    "qwen2.5-1.5b": 0.04, "qwen2.5-3b": 0.06, "qwen2.5-7b": 0.12}.get(model, 0.08)
    est_minutes = steps * min_per_step
    est_cost    = est_minutes / 60 * gpu_rate

    print(f"\n{'='*55}")
    print(f"  Model      : {MODELS[model][0]}")
    print(f"  Steps      : {steps}")
    print(f"  GPU        : {gpu.upper()} @ ~${gpu_rate:.2f}/hr")
    print(f"  Est. time  : ~{est_minutes:.0f} min")
    print(f"  Est. cost  : ~${est_cost:.2f}  (well under $25)")
    print(f"{'='*55}")

    if dry_run:
        print("\n--dry-run: exiting without launching training.")
        return

    result = _train_remote.with_options(gpu=GPU_MAP[gpu]).remote(
        model_key=model,
        max_steps=steps,
        num_train_samples=samples,
        num_generations=generations,
        learning_rate=lr,
        push_to_hub=push,
        hf_repo=repo,
        wandb_project=wandb_project,
    )
    print("\nResult:", result)
