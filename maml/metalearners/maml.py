import torch
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm

from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters
from maml.utils import tensors_to_device, compute_accuracy, gradient_update_parameters_warp
from ..model import BatchParameter, BatchLinear

__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'FOMAML']


class ModelAgnosticMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.
    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).
    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.
    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].
    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.
    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.
    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].
    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.
    device : `torch.device` instance, optional
        The device on which the model is defined.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)
    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(self, model, optimizer=None, warp_optimizer=None, warp_model=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None, warp_scheduler=None,
                 loss_function=F.cross_entropy, device=None, num_maml_steps=0, ensembler=None, ensembler_optimizer=None, ensemble_size=4, ensembler_scheduler=None):
        self.model = model.to(device=device)
        self.warp_model = warp_model.to(device=device) if warp_model is not None else None
        self.ensembler = ensembler.to(device=device) if ensembler is not None else None
        self.optimizer = optimizer
        self.warp_optimizer = warp_optimizer
        self.ensembler_optimizer = ensembler_optimizer
        self.ensemble_size = ensemble_size
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.warp_scheduler = warp_scheduler
        self.ensembler_scheduler = ensembler_scheduler
        self.loss_function = loss_function
        self.device = device
        self.num_maml_steps = num_maml_steps
        self.num_steps = 0

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if per_param_step_size else [self.step_size]})

            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs= [group['initial_lr']
                    for group in self.optimizer.param_groups]

    def get_outer_loss(self, batch, eval_mode=False, repeats=32, write_params=False):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            #'inner_losses': np.zeros((self.num_adaptation_steps,
            #    num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        mean_outer_loss = torch.tensor(0., device=self.device)

        train_inputs = batch["train"][0]
        train_targets = batch["train"][1]
        test_inputs = batch["test"][0]
        test_targets = batch["test"][1]

        if self.ensembler is not None:
            train_inputs = train_inputs.repeat_interleave(self.ensemble_size, dim=0)
            train_targets = train_targets.repeat_interleave(self.ensemble_size, dim=0)
            test_inputs = test_inputs.repeat_interleave(self.ensemble_size, dim=0)

        if False:
            for i in range(repeats):
                train_input_exp =  train_inputs[i * (train_inputs.size(0) // repeats) : (i+1) * (train_inputs.size(0) // repeats)].detach().repeat_interleave(repeats, dim=0)
                train_target_exp =  train_targets[i * (train_inputs.size(0) // repeats) : (i+1) * (train_inputs.size(0) // repeats)].repeat_interleave(repeats, dim=0)
                test_input_exp =  test_inputs[i * (train_inputs.size(0) // repeats) : (i+1) * (train_inputs.size(0) // repeats)].detach().repeat_interleave(repeats, dim=0)
                train_input_exp.requires_grad_(True)
                test_input_exp.requires_grad_(True)

                params, adaptation_results = self.adapt(train_input_exp, train_target_exp,
                    is_classification_task=is_classification_task,
                    num_adaptation_steps=self.num_adaptation_steps,
                    step_size=self.step_size, first_order=self.first_order)

                with torch.set_grad_enabled(False):
                    test_logits = self.model(test_input_exp, params=params)
                    test_logits = test_logits.view(-1, repeats, *test_logits.size()[1:])
                    test_logits = test_logits.sum(dim=1)
                    outer_loss = self.loss_function(test_logits, test_targets[i * (train_inputs.size(0) // repeats) : (i+1) * (train_inputs.size(0) // repeats)])
                    mean_outer_loss += outer_loss

                if is_classification_task:
                    results['accuracies_after'][i * (train_inputs.size(0) // repeats) : (i+1) * (train_inputs.size(0) // repeats)] = compute_accuracy(
                        test_logits, test_targets[i * (train_inputs.size(0) // repeats) : (i+1) * (train_inputs.size(0) // repeats)])

                torch.cuda.empty_cache()

            mean_outer_loss.div_(repeats)
            results['mean_outer_loss'] = mean_outer_loss.item()

        params, adaptation_results = self.adapt(train_inputs, train_targets,
            is_classification_task=is_classification_task,
            num_adaptation_steps=self.num_adaptation_steps,
            step_size=self.step_size, first_order=self.first_order, write_params=write_params)

        #results['inner_losses'][:] = adaptation_results['inner_losses']
        if is_classification_task:
            results['accuracies_before'] = adaptation_results['accuracy_before']

        with torch.set_grad_enabled(self.model.training):
            test_logits = self.model(test_inputs, params=params)
            if self.ensembler is not None:
                temp = test_logits
                temp = temp.view(-1, self.ensemble_size, *temp.size()[1:])
                temp = temp.transpose(0,1).reshape(self.ensemble_size, -1, temp.size(-1))
                temp = self.ensembler(temp)
                temp = temp.sum(dim=0)
                test_logits = temp.view(test_logits.size(0) // self.ensemble_size, test_logits.size(1), -1)

            outer_loss = self.loss_function(test_logits, test_targets)
            mean_outer_loss += outer_loss

        if is_classification_task:
            results['accuracies_after'] = compute_accuracy(
                test_logits, test_targets)

        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results

    def adapt(self, inputs, targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False, write_params=False, reset_params=False):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
        params = OrderedDict(self.model.meta_named_parameters())

        for key in params.keys():
            if isinstance(params[key], BatchParameter) and params[key].expanding:
                params[key] = params[key].expanded(inputs.size(0))

        results = {}

        state = None
        for step in range(num_adaptation_steps):
            if self.warp_model is not None:
                self.warp_model.set_listening(True)

            logits = self.model(inputs, params=params)
            inner_loss = self.loss_function(logits, targets, reduction="none")

            if inner_loss.ndim > 2:
                inner_loss = inner_loss.mean(dim=-1).sum(dim=0)
            else:
                inner_loss = inner_loss.sum(dim=0)

            if (step == 0) and is_classification_task:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            params = gradient_update_parameters_warp(self.model, inner_loss,
                warp_model=self.warp_model, step_size=step_size, params=params,
                first_order=(not self.model.training) or first_order, state=state)

            if write_params:
                old_params = OrderedDict(self.model.meta_named_parameters())
                with torch.no_grad():
                    for key in old_params.keys():
                        # old_params[key].copy_(params[key]=True)
                        old_params[key].detach_().requires_grad_(True)

            if reset_params:
                for module in self.model.modules():
                    if isinstance(module, BatchLinear):
                        module.reset_parameters()

            if self.warp_model is not None:
                self.warp_model.set_listening(False)

        return params, results

    def train(self, dataloader, max_batches=500, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                self.num_steps = 0
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                wandb_data = {'outer_loss': results['mean_outer_loss']}
                if 'accuracies_after' in results:
                    wandb_data['accuracy'] = float(np.mean(results['accuracies_after']).item())
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies_after']))
                pbar.set_postfix(**postfix)
                wandb.log(wandb_data)

    def train_iter(self, dataloader, max_batches=500):
        """
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        """
        num_batches = 0
        self.model.train()
        if self.warp_model is not None:
            self.warp_model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                self.num_steps += 1

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                if self.warp_optimizer is not None:
                    self.warp_optimizer.zero_grad()

                if self.ensembler_optimizer is not None:
                    self.ensembler_optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)
                yield results

                outer_loss.backward()


                if self.optimizer is not None and (self.num_maml_steps <= 0 or self.num_steps < self.num_maml_steps):
                    self.optimizer.step()

                if self.warp_optimizer is not None:
                    self.warp_optimizer.step()

                if self.ensembler_optimizer is not None:
                    self.ensembler_optimizer.step()

                if self.optimizer is not None:
                    for param_group in self.optimizer.param_groups:
                        wandb.log({"maml_lr": param_group['lr']}, commit=False)
                        break

                if self.warp_optimizer is not None:
                    for param_group in self.warp_optimizer.param_groups:
                        wandb.log({"warp_lr": param_group['lr']}, commit=False)
                        break

                if self.ensembler_optimizer is not None:
                    for param_group in self.ensembler_optimizer.param_groups:
                        wandb.log({"ensembler_lr": param_group['lr']}, commit=False)
                        break

                if self.scheduler is not None:
                    self.scheduler.step()

                if self.warp_scheduler is not None:
                    self.warp_scheduler.step()

                if self.ensembler_scheduler is not None:
                    self.ensembler_scheduler.step()

                """
                for module in self.model.modules():
                    if isinstance(module, BatchLinear):
                        module.reset_parameters()
                """

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        wandb_data = {
            "mean_outer_loss_eval": mean_outer_loss,
            "mean_accuracy_eval": mean_accuracy
        }
        wandb.log(wandb_data, commit=False)

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        if self.warp_model is not None:
            self.warp_model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch, eval_mode=True, write_params=False)
                yield results

                num_batches += 1

MAML = ModelAgnosticMetaLearning

class FOMAML(ModelAgnosticMetaLearning):
    def __init__(self, model, optimizer=None, step_size=0.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(FOMAML, self).__init__(model, optimizer=optimizer, first_order=True,
            step_size=step_size, learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps, scheduler=scheduler,
            loss_function=loss_function, device=device)
