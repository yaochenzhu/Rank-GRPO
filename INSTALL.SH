#!/bin/bash
TRL_PATH=$(python -c "import trl, os; print(os.path.dirname(trl.__file__))")
cp libs/trl/rank_grpo_trainer.py $TRL_PATH/trainer/
cp libs/trl/__module__init__.py $TRL_PATH/__init__.py
echo "✅ Rank-GRPO trainer installed to $TRL_PATH"