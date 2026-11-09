"""W6b trainer (results/w6b_distill/PREREG.md section 3): the bundle's `scripts.trainer.Vista3dTrainer` with ONE addition, a self-distillation term on K non-tumour classes against the frozen baseline.
The task part of an iteration is the bundle's own code, unchanged (prompt sampling, DiceCE per prompted class, AMP, optimiser step). The K distilled classes ride in the SAME student forward as extra class prompts (the class head scores every prompt independently, checked by `checks`), the teacher (a frozen copy of the baseline weights, eval mode, no gradient) is run once on the same patch for those K classes, and the loss is  task + lam * mean BCE(student logit, teacher probability)  (glassbox.seg.w6b_loss). Needs the bundle's `scripts` package on sys.path (the driver scripts/w6b_train.py does that)."""
import copy

import numpy as np
import torch
from monai.engines.utils import IterationEvents
from monai.utils import CommonKeys as Keys
from monai.apps.vista3d.sampler import sample_prompt_pairs
from scripts.trainer import Vista3dTrainer

from glassbox.seg.w6b_loss import distill_bce, sample_distill_classes


class W6bTrainer(Vista3dTrainer):
    def __init__(self, *args, w6b, **kwargs):
        super().__init__(*args, **kwargs)
        self.w6b = dict(w6b); self.teacher = None; self.w6b_gen = torch.Generator().manual_seed(int(self.w6b["seed"])); self.w6b_checks = {}; self.w6b_hist = {"task": [], "distill": [], "kl": []}

    def _ensure_teacher(self, device):
        if self.teacher is None:
            t = copy.deepcopy(self.network); t.load_state_dict(torch.load(self.w6b["weights"], map_location="cpu")); t.to(device).eval()
            for q in t.parameters():
                q.requires_grad_(False)
            self.teacher = t; self.w6b_checks["teacher_param_abs_sum_start"] = float(sum(q.detach().abs().sum().item() for q in t.parameters()))

    def _iteration(self, engine, batchdata):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        inputs, labels = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: labels}
        self._ensure_teacher(engine.state.device)
        label_set = engine.hyper_kwargs["label_set"]; output_classes = engine.hyper_kwargs["output_classes"]
        if label_set is None:
            label_set = np.arange(output_classes).tolist()
        label_prompt, point, point_label, prompt_class = sample_prompt_pairs(
            labels, label_set, image_size=engine.hyper_kwargs["patch_size"], max_point=engine.hyper_kwargs["max_point"], max_prompt=engine.hyper_kwargs["max_prompt"], max_backprompt=engine.hyper_kwargs["max_backprompt"],
            max_foreprompt=engine.hyper_kwargs["max_foreprompt"], drop_label_prob=engine.hyper_kwargs["drop_label_prob"], drop_point_prob=engine.hyper_kwargs["drop_point_prob"], include_background=not engine.hyper_kwargs["exclude_background"])
        assert label_prompt is not None and point is None, "W6b needs the class-prompt-only training path (drop_label_prob 0, drop_point_prob 1)"
        n_task = label_prompt.shape[0]; dcls = sample_distill_classes(self.w6b_gen, self.w6b["classes"], int(self.w6b["k"])); dcv = torch.tensor(dcls, device=engine.state.device).long().unsqueeze(-1)
        cv = torch.cat([label_prompt, dcv], 0); lam = float(self.w6b["lam"]); first = engine.state.iteration <= 1 and bool(self.w6b.get("checks"))

        def _compute_pred_loss():
            outputs = engine.network(input_images=inputs, point_coords=None, point_labels=None, class_vector=cv)
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            loss_task, loss_n = torch.tensor(0.0, device=engine.state.device), torch.tensor(0.0, device=engine.state.device)
            for id in range(len(prompt_class)):
                if prompt_class[id] == 0:
                    continue
                loss_task += engine.loss_function(outputs[[id]].float(), labels == prompt_class[id]); loss_n += 1.0
            loss_task /= max(loss_n, 1.0)
            with torch.no_grad():
                t_out = self.teacher(input_images=inputs, point_coords=None, point_labels=None, class_vector=dcv)
            s_out = outputs[n_task:]; l_dist, l_kl = distill_bce(s_out, t_out); loss = loss_task + lam * l_dist
            if first:      # smoke-test checks, first iteration only: (d) shapes, (b) distillation value and gradient at student == teacher, (f) the task loss is the same without the extra prompts
                ps = [q for q in engine.network.parameters() if q.requires_grad]; g = torch.autograd.grad(l_dist, ps, retain_graph=True, allow_unused=True)
                self.w6b_checks.update({"outputs_shape": list(outputs.shape), "student_distill_shape": list(s_out.shape), "teacher_shape": list(t_out.shape), "n_task_prompts": int(n_task), "distilled_classes_it1": dcls,
                                        "it1_distill_bce": float(l_dist), "it1_distill_kl": float(l_kl), "it1_distill_grad_norm": float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None))), "it1_task_loss": float(loss_task)})
                with torch.no_grad():
                    o2 = engine.network(input_images=inputs, point_coords=None, point_labels=None, class_vector=label_prompt); lt, ln = torch.tensor(0.0, device=engine.state.device), 0.0
                    for id in range(len(prompt_class)):
                        if prompt_class[id] != 0:
                            lt += engine.loss_function(o2[[id]].float(), labels == prompt_class[id]); ln += 1.0
                    self.w6b_checks["it1_task_loss_without_extra_prompts"] = float(lt / max(ln, 1.0)); del o2
            engine.state.output[Keys.LOSS] = loss; engine.state.output["loss_task"] = float(loss_task); engine.state.output["loss_distill"] = float(l_dist); engine.state.output["loss_distill_kl"] = float(l_kl)
            self.w6b_hist["task"].append(float(loss_task)); self.w6b_hist["distill"].append(float(l_dist)); self.w6b_hist["kl"].append(float(l_kl))
            outputs = None; torch.cuda.empty_cache()
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        engine.network.train(); engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
        if engine.amp and engine.scaler is not None:
            with torch.amp.autocast("cuda", **engine.amp_kwargs):
                _compute_pred_loss()
            engine.scaler.scale(engine.state.output[Keys.LOSS]).backward(); engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            engine.scaler.step(engine.optimizer); engine.scaler.update()
        else:
            _compute_pred_loss(); engine.state.output[Keys.LOSS].backward(); engine.fire_event(IterationEvents.BACKWARD_COMPLETED); engine.optimizer.step()
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        return engine.state.output
