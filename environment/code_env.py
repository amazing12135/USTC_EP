import subprocess

class CodeEnv:
    def __init__(self):
        self.success_reward = 1.0
        self.error_penalty = -0.5
        self.timeout = 5

    def execute_code(self, code: str) -> tuple[bool, str]:
        """在隔离的子进程中执行代码并捕获输出"""
        try:
            result = subprocess.run(
                ['python', '-c', code],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

    def compute_reward(self, generated_code: str, expected_output: str) -> float:
        """根据执行结果与期望输出匹配度计算奖励"""
        success, output = self.execute_code(generated_code)
        if success and output == expected_output.strip():
            return self.success_reward
        return self.error_penalty
