Periodic Intra-Ensemble Knowledge Distillation for Reinforcement Learning
=

Introduction
---
This is official implementation of an under-review conference paper titled "Periodic intra-ensemble Knowledge Distillation for Reinforcement Learning". Reinforcement Learning (RL) has demonstrated promising results across several sequential decision-making tasks. However, reinforcement learning struggles to learn efficiently, thus limiting its pervasive application to several challenging problems. A typical RL agent learns solely from its own trial-and-error experiences, requiring many experiences to learn a successful policy. To alleviate this problem, we propose *periodic intra-ensemble knowledge distillation* (PIEKD). PIEKD is a learning framework that uses an ensemble of RL agents to execute different policies in the environment while sharing knowledge amongst agents in the ensemble. Our experiments demonstrate that PIEKD improves upon state-of-the-art RL methods in sample efficiency and performance on several challenging MuJoCo benchmark tasks. Additionally, we present an in-depth investigation on how PIEKD leads to performance improvements.

Installation
---
- Ensure you use python3
- Run ```pip install -r requirement.txt```
- Install custom_chainerrl:
```
cd custom_chainerrl
pip install -e .
```

Usage
---
- Run ```bash train_piekd.sh ${environment name} (e.g., HalfCheetah-v2)```



