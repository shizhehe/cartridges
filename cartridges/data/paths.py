from pathlib import Path


MONKEYS = Path(__file__).parent.parent.parent / "data/example_docs/monkeys.txt"
MINIONS = Path(__file__).parent.parent.parent / "data/example_docs/minions.txt"
AZALIA_DPO = Path(__file__).parent.parent.parent /"data/example_docs/azalia_device_placement_optimization.txt"
AZALIA_FAST = Path(__file__).parent.parent.parent /"data/example_docs/azalia_fast_paper.txt"

AZALIA_DPO_TITLE = "Device Placement Optimization with Reinforcement Learning"
AZALIA_FAST_TITLE = "A Full-Stack Search Technique for Domain Optimized Deep Learning Accelerators"

MONKEYS_TITLE = "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"
MINIONS_TITLE = "Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models"