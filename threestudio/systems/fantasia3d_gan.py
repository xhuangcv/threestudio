from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from threestudio.systems.utils import parse_optimizer
from threestudio.utils.GAN.loss import discriminator_loss, generator_loss
from threestudio.utils.GAN.discriminator import NLayerDiscriminator, weights_init


@threestudio.register("fantasia3d-gan-system")
class Fantasia3DGAN(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        latent_steps: int = 1000
        texture: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

        self.discriminator = NLayerDiscriminator(
            input_nc=3, n_layers=3, use_actnorm=False, ndf=64
        ).apply(weights_init)
        
        self.automatic_optimization = False

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_rgb=self.cfg.texture)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        self.toggle_optimizer(optimizer_g)

        loss = 0.0

        out = self(batch)
        prompt_utils = self.prompt_processor()


        if not self.cfg.texture:  # geometry training
            if self.true_global_step < self.cfg.latent_steps:
                guidance_inp = torch.cat(
                    [out["comp_normal"] * 2.0 - 1.0, out["opacity"]], dim=-1
                )
                guidance_out = self.guidance(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=True
                )
            else:
                guidance_inp = out["comp_normal"]
                guidance_out = self.guidance(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=False
                )

            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
        else:  # texture training
            guidance_inp = out["comp_rgb"]
            if isinstance(
                self.guidance,
                threestudio.models.guidance.controlnet_guidance.ControlNetGuidance,
            ):
                cond_inp = out["comp_normal"]
                guidance_out = self.guidance(
                    guidance_inp, cond_inp, prompt_utils, **batch, rgb_as_latents=False
                )
            else:
                guidance_out = self.guidance(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=False
                )

        
        loss_G = generator_loss(
            self.discriminator,
            guidance_out["edit_images"].permute(0, 3, 1, 2),
            guidance_out["rgb_bchw"],
        )

        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        
        loss = loss + loss_G * self.C(self.cfg.loss["lambda_G"])
        self.manual_backward(loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        loss_D = discriminator_loss(
            self.discriminator,
            guidance_out["edit_images"].permute(0, 3, 1, 2),
            guidance_out["rgb_bchw"],
        )
        loss_D *= self.C(self.cfg.loss["lambda_D"])
        self.log("train/loss_D", loss_D)
        self.manual_backward(loss_D)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

    def configure_optimizers(self):
        optimizer_g = parse_optimizer(self.cfg.optimizer, self)
        optimizer_d = parse_optimizer(self.cfg.optimizer.optimizer_dis, self)
        return [optimizer_g, optimizer_d], []