# AI Partner Context: Project Phoenix - Safe & Agile Drone Control

## 1. Core Scientific Objective

Our mission is to fuse the strengths of two seminal papers to achieve the holy grail of drone flight: **provable safety with maximum efficiency**.

- **GCBF+ (MIT)**: Provides a mathematical framework for a "safety shield" via Control Barrier Functions. This is our foundation for **Safety**.
- **Back to Newton's Laws (SJTU)**: Enables end-to-end policy optimization by backpropagating gradients through a differentiable physics simulator. This is our engine for **Efficiency & Agility**.

## 2. The Core Conflict: The Safety-Efficiency Trilemma

Our entire research journey (v1-v7 experiments) has been a battle to resolve the conflict between:
- **Safety** (Low Collision Rate)
- **Efficiency** (High Success Rate, Low Timeout Rate)
- **Smoothness** (Low Jerk)

## 3. Architectural Evolution & Key Findings

- **v2 (Safety-Gated)**: Achieved **peak safety** (23.2% collision) but was overly conservative (69.8% timeout). **Finding**: A hard-gated `alpha_safety` value works for safety but paralyzes the agent.
- **v5 (Progress Reward)**: Achieved **peak efficiency** (42% success) but was reckless (37.2% collision). **Finding**: A strong goal-oriented reward creates an aggressive "F1 racer" persona.
- **v7 (Guardian Protocol)**: Our latest attempt at a hard-constraint system. **Problem**: Used a non-differentiable `torch.where` for the hard-gate. This "broke" the gradient flow, preventing the policy from learning *why* its proposed actions were unsafe. The policy didn't learn to be safe; it learned to gamble on the guardian's approval.

## 4. Current Mission: "The Phoenix Plan" - The Differentiable Guardian

We have identified the root cause of our plateau: the non-differentiable hard-gate in v7.

**Our immediate and sole objective is to re-architect the v7 "Guardian Protocol" into a fully differentiable end-to-end system.**

This involves replacing the crude `torch.where` switch with a mathematically sound **"Differentiable Safety Projection Layer"**. This new layer will not *reject* unsafe actions, but will *project* them onto the boundary of the safe set defined by GCBF. This ensures safety is a structural property, not a penalty, and the gradient provides clear, corrective feedback to the policy network.