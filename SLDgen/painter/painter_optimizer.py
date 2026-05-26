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

    def zero_grad_(self):
        """Zero gradients on the optimizer's parameter groups."""
        self.optim.zero_grad()

    def step_(self):
        """Perform an optimizer step to update parameters."""
        self.optim.step()

    def get_lr(self):
        """Return the learning rate of the first parameter group."""
        return self.optim.param_groups[0]["lr"]
