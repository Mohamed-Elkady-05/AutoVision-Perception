import argparse
import datetime
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchvision import datasets, transforms

from CNN_model import TrafficSignCNN


class CNNFeatureMapVisualizer:
    """Visualize CNN activations, filters, and class confidence."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names
        self.mean = mean
        self.std = std

    @staticmethod
    def load_model(
        checkpoint_path: Optional[str],
        num_classes: int = 43,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.nn.Module, bool, str]:
        """Load model from checkpoint; fallback to initialized model if unavailable."""
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TrafficSignCNN(num_classes=num_classes).to(device)

        if not checkpoint_path:
            return model, False, 'No checkpoint provided. Using initialized weights.'

        if not os.path.exists(checkpoint_path):
            msg = f'Checkpoint not found at {checkpoint_path}. Using initialized weights.'
            return model, False, msg

        payload = torch.load(checkpoint_path, map_location=device)

        state_dict = payload
        if isinstance(payload, dict):
            if 'state_dict' in payload:
                state_dict = payload['state_dict']
            elif 'model_state_dict' in payload:
                state_dict = payload['model_state_dict']

        model.load_state_dict(state_dict, strict=True)
        return model, True, f'Loaded checkpoint from {checkpoint_path}'

    def _to_display_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]

        img = image_tensor.detach().cpu().float().clone()
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        mean = torch.tensor(self.mean, dtype=img.dtype).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=img.dtype).view(-1, 1, 1)
        if img.shape[0] == mean.shape[0]:
            img = img * std + mean

        img = img.clamp(0.0, 1.0)
        img = img.permute(1, 2, 0).numpy()
        return img

    def _extract_feature_maps(self, image_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_batch = image_batch.to(self.device)
        with torch.no_grad():
            if hasattr(self.model, 'extract_feature_maps'):
                return self.model.extract_feature_maps(image_batch)

            activations: Dict[str, torch.Tensor] = {}
            hooks = []

            for layer_name in ('conv1', 'conv2', 'conv3'):
                if hasattr(self.model, layer_name):
                    module = getattr(self.model, layer_name)
                    hooks.append(
                        module.register_forward_hook(
                            lambda _m, _inp, out, name=layer_name: activations.__setitem__(name, out.detach())
                        )
                    )

            _ = self.model(image_batch)
            for hook in hooks:
                hook.remove()
            return activations

    def _predict_topk(self, image_batch: torch.Tensor, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        image_batch = image_batch.to(self.device)
        with torch.no_grad():
            logits = self.model(image_batch)
            probs = torch.softmax(logits, dim=1)
            k = min(top_k, probs.shape[1])
            top_probs, top_idx = torch.topk(probs, k=k, dim=1)

        return top_probs[0].detach().cpu().numpy(), top_idx[0].detach().cpu().numpy()

    def _plot_activation_grid(
        self,
        feature_map: torch.Tensor,
        layer_name: str,
        save_path: str,
        max_channels: int = 16,
    ) -> None:
        fmap = feature_map[0].detach().cpu()
        channels = min(max_channels, fmap.shape[0])
        cols = 4
        rows = int(np.ceil(channels / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.0 * rows))
        axes = np.array(axes).reshape(-1)

        for idx, ax in enumerate(axes):
            if idx < channels:
                sns.heatmap(fmap[idx].numpy(), cmap='viridis', cbar=False, ax=ax)
                ax.set_title(f'{layer_name} ch{idx}')
            ax.axis('off')

        fig.suptitle(f'Feature Maps - {layer_name}', fontsize=14)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _plot_aggregated_maps(
        self,
        image_tensor: torch.Tensor,
        feature_maps: Dict[str, torch.Tensor],
        save_path: str,
    ) -> None:
        img = self._to_display_image(image_tensor)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        for i, layer in enumerate(('conv1', 'conv2', 'conv3'), start=1):
            if layer in feature_maps:
                fmap = feature_maps[layer][0].detach().cpu()
                agg = torch.mean(fmap, dim=0).numpy()
                sns.heatmap(agg, cmap='magma', cbar=False, ax=axes[i])
                axes[i].set_title(f'{layer} mean activation')
                axes[i].axis('off')
            else:
                axes[i].axis('off')
                axes[i].set_title(f'{layer} unavailable')

        fig.suptitle('How the CNN Understands the Image (Aggregated Activations)', fontsize=14)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _plot_filters(self, save_dir: str, max_kernels: int = 16) -> None:
        for layer_name in ('conv1', 'conv2', 'conv3'):
            if not hasattr(self.model, layer_name):
                continue

            layer = getattr(self.model, layer_name)
            if not hasattr(layer, 'weight'):
                continue

            weights = layer.weight.detach().cpu()
            kernels = min(max_kernels, weights.shape[0])
            cols = 4
            rows = int(np.ceil(kernels / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.0 * rows))
            axes = np.array(axes).reshape(-1)

            for idx, ax in enumerate(axes):
                if idx < kernels:
                    kernel = weights[idx]
                    if kernel.shape[0] == 3:
                        rgb = kernel.permute(1, 2, 0).numpy()
                        rgb_min = rgb.min()
                        rgb_max = rgb.max()
                        if rgb_max > rgb_min:
                            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
                        ax.imshow(rgb)
                    else:
                        # For deeper conv layers, visualize filter strength map.
                        norm_map = torch.norm(kernel, dim=0).numpy()
                        sns.heatmap(norm_map, cmap='coolwarm', cbar=False, ax=ax)
                    ax.set_title(f'{layer_name} k{idx}')
                ax.axis('off')

            fig.suptitle(f'Learned Filters - {layer_name}', fontsize=14)
            fig.tight_layout()
            out = os.path.join(save_dir, f'filters_{layer_name}.png')
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)

    def _plot_confidence(
        self,
        top_probs: np.ndarray,
        top_idx: np.ndarray,
        save_path: str,
        true_label: Optional[int] = None,
    ) -> None:
        labels = []
        for cls_id in top_idx:
            if self.class_names and int(cls_id) < len(self.class_names):
                labels.append(self.class_names[int(cls_id)])
            else:
                labels.append(f'class_{int(cls_id)}')

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_probs, y=labels, color='#4A86E8', orient='h', ax=ax)
        pred_label = labels[0]
        pred_conf = float(top_probs[0])

        title = f'Prediction: {pred_label} ({pred_conf:.3f})'
        if true_label is not None:
            title += f' | True: {true_label}'
        ax.set_title(title)
        ax.set_xlabel('Probability')
        ax.set_ylabel('Class')
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def visualize_sample(
        self,
        image_tensor: torch.Tensor,
        sample_id: str,
        output_dir: str,
        true_label: Optional[int] = None,
        max_channels: int = 16,
        top_k: int = 5,
    ) -> None:
        if image_tensor.ndim == 3:
            image_batch = image_tensor.unsqueeze(0)
        else:
            image_batch = image_tensor

        feature_maps = self._extract_feature_maps(image_batch)

        sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
        os.makedirs(sample_dir, exist_ok=True)

        self._plot_aggregated_maps(
            image_tensor=image_batch,
            feature_maps=feature_maps,
            save_path=os.path.join(sample_dir, 'activations_aggregated.png'),
        )

        for layer in ('conv1', 'conv2', 'conv3'):
            if layer in feature_maps:
                self._plot_activation_grid(
                    feature_map=feature_maps[layer],
                    layer_name=layer,
                    save_path=os.path.join(sample_dir, f'feature_maps_{layer}.png'),
                    max_channels=max_channels,
                )

        top_probs, top_idx = self._predict_topk(image_batch, top_k=top_k)
        self._plot_confidence(
            top_probs=top_probs,
            top_idx=top_idx,
            save_path=os.path.join(sample_dir, 'prediction_confidence.png'),
            true_label=true_label,
        )

    def run_from_test_loader(
        self,
        test_loader,
        num_samples: int,
        output_dir: str,
        max_channels: int = 16,
        top_k: int = 5,
    ) -> None:
        dataset = test_loader.dataset
        if len(dataset) == 0:
            raise ValueError('Test dataset is empty.')

        self._plot_filters(output_dir)

        for idx in range(num_samples):
            sample_index = random.randrange(0, len(dataset))
            image, label = dataset[sample_index]
            self.visualize_sample(
                image_tensor=image,
                sample_id=f'{idx+1}_idx{sample_index}',
                output_dir=output_dir,
                true_label=int(label),
                max_channels=max_channels,
                top_k=top_k,
            )


def _build_test_loader(
    test_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader, test_dataset.classes


def _timestamped_dir(base_output_dir: str) -> str:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(base_output_dir, f'cnn_feature_maps_{timestamp}')
    os.makedirs(out, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='CNN feature maps, filters, and confidence visualizer')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint (.pth)')
    parser.add_argument('--test-dir', type=str, default='', help='ImageFolder-compatible test directory')
    parser.add_argument('--output-dir', type=str, default='visualization_outputs', help='Base output directory')
    parser.add_argument('--num-classes', type=int, default=43, help='Number of classes in classifier head')
    parser.add_argument('--samples', type=int, default=3, help='Random test samples to visualize')
    parser.add_argument('--top-k', type=int, default=5, help='Top-k classes for confidence bar plot')
    parser.add_argument('--image-size', type=int, default=32, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32, help='Test loader batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Dataloader workers')
    parser.add_argument('--max-channels', type=int, default=16, help='Max channels per layer in activation grid')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, loaded, msg = CNNFeatureMapVisualizer.load_model(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=device,
    )
    print(msg)
    if not loaded:
        print('Warning: continuing with fallback initialized weights.')

    output_dir = _timestamped_dir(args.output_dir)
    visualizer = CNNFeatureMapVisualizer(model=model, device=device, class_names=None, mean=mean, std=std)

    if args.test_dir and os.path.isdir(args.test_dir):
        test_loader, class_names = _build_test_loader(
            test_dir=args.test_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mean=mean,
            std=std,
        )
        visualizer.class_names = class_names
        visualizer.run_from_test_loader(
            test_loader=test_loader,
            num_samples=args.samples,
            output_dir=output_dir,
            max_channels=args.max_channels,
            top_k=args.top_k,
        )
        print(f'Done. Saved visualization outputs to: {output_dir}')
        return

    print('No valid --test-dir supplied. Running synthetic fallback sample.')
    visualizer._plot_filters(output_dir)
    synthetic = torch.rand(1, 3, args.image_size, args.image_size)
    visualizer.visualize_sample(
        image_tensor=synthetic,
        sample_id='synthetic_1',
        output_dir=output_dir,
        true_label=None,
        max_channels=args.max_channels,
        top_k=args.top_k,
    )
    print(f'Done. Saved visualization outputs to: {output_dir}')


if __name__ == '__main__':
    main()
