import torch
import torch.nn as nn 

from diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from transformers import AutoModelForImageClassification


class Args:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=1000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("cifar10/cifar10_uncond_50M_500K.pt")
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 

        classifier = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
        classifier.eval().cuda()

        self.classifier = classifier

    def forward(self, x, t):
        x_in = x * 2 -1
        imgs = self.sampling(x_in, t)

        imgs = torch.nn.functional.interpolate(imgs, (224, 224), mode='bicubic', antialias=True)

        with torch.no_grad():
            out = self.classifier(imgs)

        return out.logits

    def sampling(self, x, t):
        t_batch = torch.tensor([t] * len(x)).cuda()

        noise = torch.randn_like(x)

        x_t = self.diffusion.q_sample(x_start=x, t=t_batch, noise=noise)

        with torch.no_grad():

            out = self.diffusion.ddim_sample(
                self.model,
                x_t,
                t_batch,
                clip_denoised=True,
                eta=0.0
            )['pred_xstart']
                
                
        from torchvision.utils import save_image

        output_dir = "generated_samples/"
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, sample in enumerate(out):
            save_image(sample, os.path.join(output_dir, f"sample_{i}.png"))


        return out
