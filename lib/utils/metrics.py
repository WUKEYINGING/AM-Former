import numpy as np


class TrainingMetrics:
    def __init__(self):
        self.epoch_metrics = {
            'train_loss': [],
            'val_nme': [],
            'learning_rate': [],
            'grad_norm': [],
            'memory_utilization': [],
            'expert_usage': [],
            'gate_entropy': []
        }
        self.best_metrics = {
            'best_nme': float('inf'),
            'best_epoch': 0,
            'convergence_epoch': 0
        }

    def update_epoch(self, epoch, train_loss, val_nmes, current_lr, grad_norm=None):
        self.epoch_metrics['train_loss'].append(train_loss)
        self.epoch_metrics['val_nme'].append(np.mean(list(val_nmes.values())))
        self.epoch_metrics['learning_rate'].append(current_lr)
        if grad_norm:
            self.epoch_metrics['grad_norm'].append(grad_norm)

        current_avg_nme = np.mean(list(val_nmes.values()))
        if current_avg_nme < self.best_metrics['best_nme']:
            self.best_metrics['best_nme'] = current_avg_nme
            self.best_metrics['best_epoch'] = epoch

    def calculate_stability_metrics(self):
        if len(self.epoch_metrics['train_loss']) < 2:
            return {}

        losses = self.epoch_metrics['train_loss']
        nmes = self.epoch_metrics['val_nme']

        loss_volatility = np.std(losses[-10:]) if len(losses) >= 10 else np.std(losses)

        if len(nmes) >= 10:
            recent_nmes = nmes[-10:]
            nme_convergence = np.std(recent_nmes) / (np.mean(recent_nmes) + 1e-8)
        else:
            nme_convergence = np.std(nmes) / (np.mean(nmes) + 1e-8)

        return {
            'loss_volatility': loss_volatility,
            'nme_convergence_stability': nme_convergence,
            'training_consistency': 1.0 / (loss_volatility + 1e-8)
        }

    def get_complexity_report(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        module_params = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                module_params[name] = sum(p.numel() for p in module.parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'parameters_in_millions': total_params / 1e6,
            'module_breakdown': module_params
        }

    def generate_final_report(self, model, total_training_time):
        stability_metrics = self.calculate_stability_metrics()
        complexity_report = self.get_complexity_report(model)

        final_report = {
            'training_duration': total_training_time,
            'total_epochs': len(self.epoch_metrics['train_loss']),
            'best_performance': {
                'nme': self.best_metrics['best_nme'],
                'epoch': self.best_metrics['best_epoch']
            },
            'stability_metrics': stability_metrics,
            'complexity_metrics': complexity_report,
            'learning_curve': {
                'final_loss': self.epoch_metrics['train_loss'][-1],
                'final_nme': self.epoch_metrics['val_nme'][-1],
                'loss_trend': 'decreasing' if self.epoch_metrics['train_loss'][-1] < self.epoch_metrics['train_loss'][0] else 'increasing'
            }
        }

        return final_report