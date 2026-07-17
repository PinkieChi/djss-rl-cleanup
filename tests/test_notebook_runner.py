import unittest
from pathlib import Path

from djss_rl.notebook_runner import execute_notebook


class NotebookRunnerSmokeTest(unittest.TestCase):
    def test_smoke_loads_restored_dataset(self):
        project_dir = Path(__file__).resolve().parents[1]
        summary = execute_notebook(project_dir)

        self.assertEqual(summary.jobs, 50)
        self.assertEqual(summary.machines, 15)
        self.assertEqual(summary.operations, 406)
        self.assertEqual(summary.observation_shape, (14,))
        self.assertEqual(summary.action_space, 9)
        self.assertEqual(summary.costs, [])


if __name__ == "__main__":
    unittest.main()
