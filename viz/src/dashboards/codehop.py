import wandb

from dashboards.base import Dashboard, registry



registry.register(Dashboard(
    name="CodeHop",
    filters={"tags": {"$in": ["codehop"]}, 'name': {'$regex': 'wxmgty6o'}},
    table="generate_codehop/table",
    score_metric="generate_codehop/score",
    step="train/optimizer_step",
))
    