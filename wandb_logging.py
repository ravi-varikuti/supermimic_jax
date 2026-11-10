"""Optional host-side W&B logging; no SDK import when logging is disabled."""

import os


class WandbLogger:
    def __init__(self, run_dir, metadata, *, enabled=False, project=None, entity=None,
                 name=None, mode=None):
        self.run = None
        self.identity = metadata.get("wandb")
        if mode == "disabled" or not (enabled or self.identity):
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B logging requires: pip install wandb") from error
        previous = self.identity or {}
        for field, value in (("project", project), ("entity", entity)):
            if previous.get(field) and value and previous[field] != value:
                raise ValueError(f"Cannot change W&B {field} while resuming the saved run")
        options = dict(project=project or previous.get("project") or "supermimic",
                       entity=entity or previous.get("entity"), name=name or previous.get("name"),
                       dir=str(run_dir), config=metadata["config"], save_code=False)
        if mode is not None:
            options["mode"] = mode
        if previous:
            options["id"] = previous["id"]
            # Offline sessions are separate local segments; the SDK cannot resume them.
            if (mode or os.environ.get("WANDB_MODE", "online")) != "offline":
                options["resume"] = "allow"
        self.run = wandb.init(**options)
        if self.run.disabled:
            self.run = None
            return
        self.identity = {"id": self.run.id, "project": self.run.project,
                         "entity": self.run.entity, "name": self.run.name}
        self.run.define_metric("iteration")
        self.run.define_metric("*", step_metric="iteration")

    def log(self, metrics, evaluation=None):
        if self.run is None:
            return
        payload = {"iteration": metrics["iteration"]}
        for name, value in metrics.items():
            if name in ("iteration", "eval_return", "eval_failures") or value is None:
                continue
            payload["simulator_steps" if name == "simulator_steps" else f"train/{name}"] = value
        if evaluation is not None:
            import wandb
            for name in ("num_motions", "failures", "mean_normalized_return"):
                payload[f"eval/{name}"] = evaluation[name]
            rows = evaluation["clips"]
            if rows:
                columns = list(rows[0])
                payload["eval/clips"] = wandb.Table(columns=columns,
                                                   data=[[row.get(column) for column in columns] for row in rows])
            self.run.summary.update({f"eval/{name}": evaluation[name]
                                     for name in ("num_motions", "failures", "mean_normalized_return")})
        self.run.log(payload)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
