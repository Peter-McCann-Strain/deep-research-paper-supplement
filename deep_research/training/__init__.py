"""deep_research.training — RL / fine-tuning support code.

E10 noise-RL (Paper 4) lives here:
  * e10_reward_noise.py     — seeded, CPU, unit-testable judge-reward noise layer
  * e10_objective_endpoint.py — judge-free anti-Goodhart objective metric

Nothing in this package trains, calls a paid API, or mutates the canonical store
at import time. The trainer launcher is scripts/train_e10_noise_rl.py.
"""
