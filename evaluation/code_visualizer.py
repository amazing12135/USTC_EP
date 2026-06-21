import matplotlib.pyplot as plt
import os


class CodePerformanceVisualizer:
    def __init__(self, save_dir: str = 'results/figures'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_execution_accuracy(self, model_stages: list[str], success_rates: list[float],
                                filename: str = 'code_execution_accuracy.png'):
        """绘制不同模型阶段下代码执行成功率的柱状图"""
        plt.figure(figsize=(8, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = plt.bar(model_stages, success_rates, color=colors[:len(model_stages)])

        plt.ylabel('Execution Success Rate (Pass@1)')
        plt.title('Agent Code Execution Performance Comparison')
        plt.ylim(0, max(success_rates) * 1.2 if max(success_rates) > 0 else 1.0)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval * 100:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_error_rates_trend(self, epochs: list[int], syntax_errors: list[float], logic_errors: list[float],
                               filename: str = 'code_error_trend.png'):
        """绘制RL训练过程中不同错误类型比率下降的折线图"""
        plt.figure(figsize=(10, 6))

        plt.plot(epochs, syntax_errors, label='Syntax Error Rate', marker='o', linestyle='-', color='#d62728')
        plt.plot(epochs, logic_errors, label='Logic Error Rate', marker='s', linestyle='--', color='#ff7f0e')

        plt.xlabel('Training Epochs')
        plt.ylabel('Error Rate')
        plt.title('Code Generation Error Rates Trend During RL Training')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
