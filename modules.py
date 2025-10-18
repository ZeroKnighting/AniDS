import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import mse_loss, l1_loss

from pytorch_lightning import LightningModule
from nets.equiformer_md17_dens_vae import Equiformer_MD17_DeNS_VAE

class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)

class EqV2_AniDS(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None,test_type="AniDS",fix_encoder_parameters=True):
        super(EqV2_AniDS, self).__init__()
        self.save_hyperparameters(hparams)

        self.model = Equiformer_MD17_DeNS_VAE(
            irreps_in=hparams.irreps_in,
            irreps_equivariant_inputs=hparams.irreps_equivariant_inputs,   # for encoding forces during denoising positions
            irreps_node_embedding=hparams.irreps_node_embedding,
            num_layers=hparams.num_layers,
            irreps_node_attr=hparams.irreps_node_attr,
            irreps_sh=hparams.irreps_sh,
            max_radius=hparams.max_radius,
            number_of_basis=hparams.number_of_basis,
            basis_type=hparams.basis_type,
            fc_neurons=hparams.fc_neurons,
            irreps_feature=hparams.irreps_feature,        # increase numbers of channels by 4 times
            irreps_head=hparams.irreps_head,
            num_heads=hparams.num_heads,
            irreps_pre_attn=hparams.irreps_pre_attn,
            rescale_degree=hparams.rescale_degree,
            nonlinear_message=hparams.nonlinear_message,
            irreps_mlp_mid=hparams.irreps_mlp_mid,
            norm_layer=hparams.norm_layer,
            alpha_drop=hparams.alpha_drop,
            proj_drop=hparams.proj_drop,
            out_drop=hparams.out_drop,
            drop_path_rate=hparams.drop_path_rate,
            pretraining=True,
            test_type=test_type,
            fix_encoder_parameters=fix_encoder_parameters,
        )

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()
        self.criterion = L2MAELoss()


    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return [optimizer], [lr_scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """Custom learning rate scheduler step to handle both types of schedulers."""
        if self.hparams.lr_schedule == 'cosine':
            # For cosine scheduler, step on every step (not epoch)
            scheduler.step()
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
            # For reduce_on_plateau, step with metric (usually val_loss)
            if metric is not None:
                scheduler.step(metric)
        else:
            # Default behavior
            scheduler.step()

    def forward(self, batch):
        return self.model(data=batch)

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            batch["std"]=0.1
            batch["prob"]=1.0
            batch["corrupt_ratio"]=1.0
            energy_outputs, vector_outputs, kl_loss,data = self(batch)

        denoising_is_on = True
        vector_outputs = vector_outputs + energy_outputs.sum()*0

        loss_y, loss_dy, loss_pos = 0, 0, 0
        loss_e = 0.0
        loss_f = 0.0
        loss_pos = self.criterion(
            vector_outputs[(batch.noise_mask)], 
            batch.noise_vec[(batch.noise_mask)] / batch["std"]
        )

        self.losses[stage + "_pos"].append(loss_pos.detach())
        self.losses[stage + "_kl"].append(kl_loss.detach())
        # if self.hparams.derivative:
        #     if "y" not in batch:
        #         # "use" both outputs of the model's forward function but discard the first
        #         # to only use the derivative and avoid 'Expected to have finished reduction
        #         # in the prior iteration before starting a new one.', which otherwise get's
        #         # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
        #         deriv = deriv + pred.sum() * 0

        #     # force/derivative loss
        #     loss_dy = loss_fn(deriv, batch.dy)

        #     if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
        #         if self.ema[stage + "_dy"] is None:
        #             self.ema[stage + "_dy"] = loss_dy.detach()
        #         # apply exponential smoothing over batches to dy
        #         loss_dy = (
        #             self.hparams.ema_alpha_dy * loss_dy
        #             + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
        #         )
        #         self.ema[stage + "_dy"] = loss_dy.detach()

        #     if self.hparams.force_weight > 0:
        #         self.losses[stage + "_dy"].append(loss_dy.detach())

        # if "y" in batch:
        #     if (noise_pred is not None) and not denoising_is_on:
        #         # "use" both outputs of the model's forward (see comment above).
        #         pred = pred + noise_pred.sum() * 0

        #     if batch.y.ndim == 1:
        #         batch.y = batch.y.unsqueeze(1)

        #     # energy/prediction loss
        #     loss_y = loss_fn(pred, batch.y)

        #     if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
        #         if self.ema[stage + "_y"] is None:
        #             self.ema[stage + "_y"] = loss_y.detach()
        #         # apply exponential smoothing over batches to y
        #         loss_y = (
        #             self.hparams.ema_alpha_y * loss_y
        #             + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
        #         )
        #         self.ema[stage + "_y"] = loss_y.detach()

        #     if self.hparams.energy_weight > 0:
        #         self.losses[stage + "_y"].append(loss_y.detach())

        # total loss
        loss = loss_pos * self.hparams.denoising_weight +kl_loss*1#kl_weight

        self.losses[stage].append(loss.detach())

        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith("train") and len(v) > 0)}
            train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step   
            train_metrics['batch_pos_mean'] = batch.pos.mean().item()
            self.log_dict(train_metrics, sync_dist=True)

        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        should_reset = (
            self.current_epoch % self.hparams.test_interval == 0
            or (self.current_epoch - 1) % self.hparams.test_interval == 0
        )
        if should_reset:
            # reset validation dataloaders before and after testing epoch, which is faster
            # than skipping test validation steps by returning None
            self.trainer.reset_val_dataloader(self)

    # TODO(shehzaidi): clean up this function, redundant logging if dy loss exists.
    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(
                    self.losses["train_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                    result_dict["test_loss_dy"] = torch.stack(
                        self.losses["test_dy"]
                    ).mean()

            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses['val_y']) > 0:
              result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean()

            # if denoising is present, also log it
            if len(self.losses["train_pos"]) > 0:
                result_dict["train_loss_pos"] = torch.stack(
                    self.losses["train_pos"]
                ).mean()

            if len(self.losses["val_pos"]) > 0:
                result_dict["val_loss_pos"] = torch.stack(
                    self.losses["val_pos"]
                ).mean()

            if len(self.losses["test_pos"]) > 0:
                result_dict["test_loss_pos"] = torch.stack(
                    self.losses["test_pos"]
                ).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_dy": [],
            "val_dy": [],
            "test_dy": [],
            "train_pos": [],
            "val_pos": [],
            "test_pos": [],
            "train_kl": [],
            "val_kl": [],
            "test_kl": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}