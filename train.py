import click
import math
import torch.cuda
from scipy import stats
from torch import nn
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

from dataset import JadeDataLoader
from model import JadeQualityPredictorTwoStream
from utils import load_yaml, select_device, AvgLoss, ResultSaver

model_dict = {
    "resnet18": {
        "model": "models.resnet18(pretrained=True)",
        "hook": ["layer1", "layer2", "layer3", "layer4"],
        "pool": True,
        "out_channels": [64, 128, 256, 512]
    },
    "wide_resnet50": {
        "model": "models.wide_resnet50_2()",
        "hook": ["layer1", "layer2", "layer3", "layer4"],
        "pool": True,
        "out_channels": [256, 512, 1024, 2048]
    },
    # "densenet": {
    #     "model": "models.densenet201()",
    #     "hook": ["norm5"],
    #     "pool": True,
    #     "out_channels": [1920]
    # },
    # "efficientnet": {
    #     "model": "timm.create_model('efficientnet_b6')",
    #     "hook": ["blocks.6.2"],
    #     "pool": True,
    #     "out_channels": [576]
    # },
    # "vit": {
    #     "model": "timm.create_model('vit_base_patch16_224')",
    #     "hook": ["head_drop"],
    #     "pool": False,
    #     "out_channels": [768]
    # },
    "swin": {
        "model": "timm.create_model('swin_base_patch4_window7_224')",
        "hook": ["stages.0", "stages.1", "stages.2", "stages.3"],
        "pool": True,
        "out_channels": [128, 256, 512, 1024]
    },
    "swinv2": {
        "model": "timm.create_model('swinv2_cr_base_224')",
        "hook": ["stages.0", "stages.1", "stages.2", "stages.3"],
        "pool": True,
        "out_channels": [128, 256, 512, 1024]
    },
    "swinv2_t": {
        "model": "timm.create_model('swinv2_cr_tiny_ns_224.sw_in1k', pretrained=True)",
        "hook": ["stages.0", "stages.1", "stages.2", "stages.3"],
        "pool": True,
        "out_channels": [96, 192, 384, 768]
    }
}


class Trainer:
    def __init__(self, arg, device, dataloader):
        self.device = device
        self.arg = arg
        self.dataloader = dataloader
        self.model = JadeQualityPredictorTwoStream(model_dict[arg["model"][0]], model_dict[arg["model"][1]]).to(self.device)
        self.task_log_scales = nn.Parameter(torch.zeros(2, device=self.device))
        decay = float(arg.get("ema_decay", 0.999))
        self.ema = AveragedModel(self.model, avg_fn=lambda avg, cur, n: decay * avg + (1.0 - decay) * cur)
        self._make_optimizer(backbones=False)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.best_ccc = -float("inf")
        self.bad_epochs = 0
        self.mae_saver = ResultSaver(f"{arg['model']}_fold{arg['fold_index']}")

    def _make_optimizer(self, backbones):
        backbone_param_ids = {id(p) for wrapper in (self.model.backbone, self.model.backbone2)
                              for p in wrapper.parameters()}
        norm_param_ids = {id(p) for wrapper in (self.model.backbone, self.model.backbone2)
                          for module in wrapper.modules() if isinstance(module, nn.LayerNorm)
                          for p in module.parameters()}
        head_params = [p for p in self.model.parameters()
                       if p.requires_grad and (id(p) not in backbone_param_ids or id(p) in norm_param_ids)]
        head_params.append(self.task_log_scales)
        groups = [{"params": head_params, "lr": self.arg["learning_rate"], "base_lr": self.arg["learning_rate"]}]
        if backbones:
            bb_params = [p for p in self.model.parameters()
                         if p.requires_grad and id(p) in backbone_param_ids and id(p) not in norm_param_ids]
            if bb_params:
                bb_lr = self.arg.get("backbone_learning_rate", self.arg["learning_rate"] / 10)
                groups.append({"params": bb_params, "lr": bb_lr, "base_lr": bb_lr})
        self.optim = AdamW(groups, weight_decay=self.arg.get("weight_decay", 0.05))


    def train(self):
        for e in range(self.arg["epoch"]):
            if e == self.arg.get("freeze_epochs", 15):
                self.model.unfreeze_last_stages(self.arg.get("unfreeze_stages", 1))
                self._make_optimizer(backbones=True)
            warmup_epochs = self.arg.get("warmup_epochs", 5)
            progress = min(1.0, (e + 1) / max(1, warmup_epochs))
            progress *= 0.5 * (1.0 + math.cos(math.pi * e / max(1, self.arg["epoch"])))
            for group in self.optim.param_groups:
                group["lr"] = group["base_lr"] * progress
            self.model.train()
            if e < self.arg.get("freeze_epochs", 15):
                self.model.backbone.eval()
                self.model.backbone2.eval()
            avg_loss = AvgLoss()
            with tqdm(
                    self.dataloader["train"], desc=f"[epoch {e}]"
            ) as data_iterator:
                for data in data_iterator:
                    images = data.get("image_color", data["image"]).to(self.device)
                    texture_images = data.get("image_texture", data["image"]).to(self.device)
                    color_labels = data["color_score"].float().to(self.device)
                    texture_labels = data["texture_score"].float().to(self.device)
                    color_var = data["color_variance"].float().to(self.device)
                    texture_var = data["texture_variance"].float().to(self.device)
                    self.optim.zero_grad(set_to_none=True)
                    sigma_freeze = e < self.arg.get("sigma_warmup_epochs", 5)
                    with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                        out = self.model(images, texture_images)
                        lc = out["color_logvar"].flatten().clamp(self.arg.get("logvar_min", -6), self.arg.get("logvar_max", 2))
                        lt = out["texture_logvar"].flatten().clamp(self.arg.get("logvar_min", -6), self.arg.get("logvar_max", 2))
                        if sigma_freeze:
                            lc, lt = lc.detach(), lt.detach()
                        color_loss = 0.5 * (lc + (color_labels - out["color"].flatten()) ** 2 * torch.exp(-lc)).mean()
                        texture_loss = 0.5 * (lt + (texture_labels - out["texture"].flatten()) ** 2 * torch.exp(-lt)).mean()
                        if self.arg.get("boundary_weight", 0) > 0:
                            bw = self.arg["boundary_weight"]
                            color_loss += bw * (torch.relu(2.2 - out["color"].flatten()) ** 2 + torch.relu(out["color"].flatten() - 8.8) ** 2).mean()
                            texture_loss += bw * (torch.relu(2.2 - out["texture"].flatten()) ** 2 + torch.relu(out["texture"].flatten() - 8.8) ** 2).mean()
                        if self.arg.get("use_expert_variance", True) and not sigma_freeze:
                            lam = self.arg.get("expert_variance_weight", 0.5)
                            for pred_lv, expert_var in ((lc, color_var), (lt, texture_var)):
                                valid = expert_var > 0
                                if valid.any():
                                    target_lv = expert_var[valid].clamp_min(1e-4).log()
                                    variance_loss = nn.functional.mse_loss(pred_lv[valid], target_lv)
                                    if pred_lv is lc:
                                        color_loss += lam * variance_loss
                                    else:
                                        texture_loss += lam * variance_loss
                        scales = self.task_log_scales.clamp(-3, 3)
                        loss = torch.exp(-scales[0]) * color_loss + scales[0] + torch.exp(-scales[1]) * texture_loss + scales[1]
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + [self.task_log_scales], self.arg.get("grad_clip", 1.0))
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.ema.update_parameters(self.model)
                    avg_loss.update(loss.item())
                    data_iterator.set_postfix({"avg_Loss": avg_loss.average()})
            # if (e + 1) % 10 == 0:
            color_mae, texture_mae = self.test(verbose=False)
            self.mae_saver.save(e, color_mae, texture_mae)
            score = getattr(self, "last_mean_ccc", -float("inf"))
            if score > self.best_ccc:
                self.best_ccc, self.bad_epochs = score, 0
            else:
                self.bad_epochs += 1
            if self.bad_epochs >= self.arg.get("patience", 12):
                print(f"Early stopping at epoch {e + 1}; best mean CCC={self.best_ccc:.4f}")
                break
            # self.random_test()

    def test(self, verbose=True):
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                    self.dataloader["test"], desc=f"[testing]", leave=False
            ) as data_iterator:
                color_scores = list()
                texture_scores = list()
                color_labels = list()
                texture_labels = list()
                for data in data_iterator:
                    images = data.get("image_color", data["image"]).to(self.device)
                    texture_images = data.get("image_texture", data["image"]).to(self.device)
                    color_label = data["color_score"].float().to(self.device)
                    texture_label = data["texture_score"].float().to(self.device)
                    out = self.ema(images, texture_images)
                    # predicted = torch.max(out, dim=1).indices
                    # correct += (predicted == labels).sum().item()

                    color_scores.append(out["color"])
                    texture_scores.append(out["texture"])
                    color_labels.append(color_label)
                    texture_labels.append(texture_label)
            color_scores = torch.cat(color_scores).flatten().cpu()
            texture_scores = torch.cat(texture_scores).flatten().cpu()
            color_labels = torch.cat(color_labels).cpu()
            texture_labels = torch.cat(texture_labels).cpu()

            color_ccc = self.compute_metric(color_scores, color_labels)
            texture_ccc = self.compute_metric(texture_scores, texture_labels)
            self.last_mean_ccc = (color_ccc + texture_ccc) / 2

            color_mae, texture_mae = torch.abs(color_scores - color_labels).mean(), torch.abs(texture_scores - texture_labels).mean()

        if verbose:
            self.mae_saver.save(0, color_mae, texture_mae)
        return color_mae, texture_mae

    def compute_metric(self, pred, label):
        SROCC = stats.spearmanr(label, pred)[0]
        KROCC = stats.kendalltau(label, pred)[0]
        PLCC = stats.pearsonr(label, pred)[0]
        RMSE = torch.sqrt(((label - pred) ** 2).mean())
        print(f"PLCC: {PLCC:.2f} - "
              f"KROCC: {KROCC:.2f} - "
              f"SROCC: {SROCC:.2f} - "
              f"RMSE: {RMSE.item():.2f}")
        p, y = pred.float(), label.float()
        ccc = 2 * ((p - p.mean()) * (y - y.mean())).mean() / (p.var(unbiased=False) + y.var(unbiased=False) + (p.mean() - y.mean()) ** 2 + 1e-8)
        return ccc.item()

    # def random_test(self):
    #     total = self.dataset["test"].len
    #     random_sample = self.dataset["test"].__getitem__(random.randint(0, total - 1))
    #     out = self.model(random_sample["image"].unsqueeze(0).to(self.device))
    #     for key in ["Color", "Texture"]:
    #         print(f"Sample {random_sample['name']}  {key} Score : "
    #               "Predict ", colorstr('red', 'bold', f"{out[key.lower()].item() :.2f}"), ", "
    #               "Real ", colorstr('red', 'bold', f"{random_sample[f'{key.lower()}_score'] :.2f}"))


@click.command()
@click.option("--device", type=str, default="cpu")
@click.option("--config", "-c", type=str, required=True)
@click.option("--batch_size", "-bs", type=int, default=None)
@click.option("--source", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--score_file_name", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--epoch", "-e", type=int, default=None)
@click.option("--image_size", type=int, default=None)
@click.option("--learning_rate", "-lr", type=float, default=None)
@click.option("--train/--test", default=True)
def main(**kwargs):
    config = load_yaml(kwargs["config"])
    arg = kwargs.copy()
    for k, v in config.items():
        if k in arg.keys():
            if arg[k] is None or (isinstance(arg[k], tuple) and len(arg[k]) == 0):
                arg[k] = v
        else:
            arg[k] = v
    device = select_device(arg["device"])
    loader = JadeDataLoader(**arg)
    # for model in ["wide_resnet50", "densenet", "efficientnet", "vit", "swin", "swinv2"]:
    # for model in ["swinv2"]:
    #     arg["model"] = model
    #     for fold_index in range(1, 6):
    #         arg["fold_index"] = fold_index
    #         print(f"{model}_fold{fold_index}_____________________")
    #         trainer = Trainer(arg, device, loader.get_loader(fold_index=fold_index))
    #         if arg["train"]:
    #             trainer.train()
    trainer = Trainer(arg, device, loader.get_loader(fold_index=arg["fold_index"]))
    if arg["train"]:
        trainer.train()


if __name__ == '__main__':
    main()
