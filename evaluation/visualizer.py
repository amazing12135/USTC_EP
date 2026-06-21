import matplotlib.pyplot as plt
import os


class PerformanceVisualizer:
    def __init__(self, save_dir: str = 'results/figures'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_learning_curve(self, steps: list[int], rewards: list[float], losses: list[float],
                            filename: str = 'learning_curve.png'):
        """绘制RL训练过程的学习曲线（包含奖励和损失）"""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Training Steps / Epochs')
        ax1.set_ylabel('Average Reward', color='tab:blue')
        ax1.plot(steps, rewards, color='tab:blue', label='Reward', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # 实例化共享相同x轴的第二个轴，用于绘制Loss
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='tab:red')
        ax2.plot(steps, losses, color='tab:red', label='Loss', linestyle='--', marker='x')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title('RL Training Performance (Reward & Loss)')
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_accuracy_comparison(self, model_stages: list[str], accuracies: list[float],
                                 filename: str = 'accuracy_comparison.png'):
        """绘制不同训练阶段模型准确率的柱状图"""
        plt.figure(figsize=(8, 6))
        # 默认提供几个不同的颜色区分
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(model_stages, accuracies, color=colors[:len(model_stages)])

        plt.ylabel('Pass@1 Accuracy')
        plt.title('Mathematics Reasoning Performance Comparison')
        plt.ylim(0, max(accuracies) * 1.2 if max(accuracies) > 0 else 1.0)

        # 在柱子上显示具体数值百分比
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval * 100:.2f}%', ha='center', va='bottom')

        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()


# 示例用法
if __name__ == '__main__':
    visualizer = PerformanceVisualizer()

    # 模拟生成训练曲线图表
    mock_steps = [10, 20, 30, 40, 50]
    mock_rewards = [0.2, 0.45, 0.6, 0.75, 0.85]
    mock_losses = [2.1, 1.5, 1.1, 0.8, 0.4]
    visualizer.plot_learning_curve(mock_steps, mock_rewards, mock_losses)

    # 模拟生成不同阶段性能对比图表
    stages = ['Base Model', 'SFT', 'RL (GRPO)']
    acc_scores = [0.15, 0.55, 0.82]
    visualizer.plot_accuracy_comparison(stages, acc_scores)
