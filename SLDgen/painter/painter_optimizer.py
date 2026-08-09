import torch


class PainterOptimizer:
    """Manage optimizer creation and stepping for a renderer."""

    def __init__(self, args, renderer):
        self.renderer = renderer
        self.args = args

    def init_optimizers(self):
        """Build optimizer param groups from renderer infos and instantiate optimizer."""
        param_infos = self.renderer.parameters()
        # create per-parameter-group dicts with scaled learning rates
        param_groups = [
            {
                "params": [info["params"][0]],
                "name": info["name"],
                "lr": self.args.lr * info["lr_ratio"],
            }
            for info in param_infos
        ]

        # instantiate optimizer based on configured name
        self.optim = torch.optim.Adam(param_groups, betas=(0.9, 0.9), eps=1e-6)

    def param_group_names(self):
        """Names of the current parameter groups, in order."""
        return [group["name"] for group in self.optim.param_groups]

    def state_dict(self):
        """Adam's state, for checkpointing.

        Adam is configured with ``betas=(0.9, 0.9)`` -- an unusually short
        second-moment memory, so losing this would not be catastrophic -- but the
        step counter drives bias correction and the payload is tiny.
        """
        return self.optim.state_dict()

    def load_state_dict(self, state_dict, param_group_names=None):
        """Restore Adam's state, refusing a checkpoint from a different param layout.

        The parameter groups depend on --optimize-cp-weights and --width optim.
        A mismatch is already impossible once the structural fingerprint has been
        checked; this is defence in depth, because loading Adam state positionally
        onto the wrong tensors would corrupt the run silently.
        """
        if param_group_names is not None:
            current = self.param_group_names()
            if list(param_group_names) != current:
                raise ValueError(
                    f"checkpoint has optimizer parameter groups {list(param_group_names)} "
                    f"but this run has {current}."
                )
        self.optim.load_state_dict(state_dict)

    def zero_grad_(self):
        """Zero gradients on the optimizer's parameter groups."""
        self.optim.zero_grad()

    def step_(self):
        """Perform an optimizer step to update parameters."""
        self.optim.step()

    def get_lr(self):
        """Return the learning rate of the first parameter group."""
        return self.optim.param_groups[0]["lr"]
