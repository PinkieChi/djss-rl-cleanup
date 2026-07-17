import unittest
from pathlib import Path

from djss_rl.environment import make_env
from djss_rl.evaluation import evaluate_checkpoint, run_scheduling
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


class ExtractedModuleSmokeTest(unittest.TestCase):
    def setUp(self):
        self.project_dir = Path(__file__).resolve().parents[1]
        self.dataset_path = self.project_dir / "Dataset 50_0.5_0.02.ini"
        self.checkpoint_path = (
            self.project_dir
            / "Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth"
        )

    def test_make_env_loads_restored_dataset(self):
        env = make_env(dataset_path=self.dataset_path)

        self.assertEqual(len(env.world.jobs), 50)
        self.assertEqual(len(env.world.machines), 15)
        self.assertEqual(env.world.operations, 406)
        self.assertEqual(env.observation_space.shape, (14,))
        self.assertEqual(env.action_space.n, 9)

    def test_reset_and_step_keep_environment_consistent(self):
        env = make_env(dataset_path=self.dataset_path)
        observation, info = env.reset()
        world = env.world

        self.assertEqual(observation.shape, (14,))
        self.assertEqual(info, {})

        world.ready_machine = next(machine for machine in world.machines if machine.request)
        env.get_legal_actions(world.ready_machine)
        next_observation, reward, done, truncated, info = env.step(2)

        self.assertEqual(next_observation.shape, (14,))
        self.assertIsInstance(float(reward), float)
        self.assertFalse(truncated)
        self.assertEqual(info, {})
        self.assertIsInstance(done, bool)
        self.assertGreaterEqual(len(world.operations_done), 0)

    def test_single_baseline_scheduling_completes(self):
        env = make_env(dataset_path=self.dataset_path)
        result = run_scheduling(env, env.world, name="SPT_DR_O", decision_rule=2)

        self.assertEqual(result.name, "SPT_DR_O")
        self.assertAlmostEqual(result.tardiness_rate, 0.6206896551724138)
        self.assertEqual(result.makespan, 2754)

    def test_saved_checkpoint_evaluates(self):
        result = evaluate_checkpoint(dataset_path=self.dataset_path, checkpoint_path=self.checkpoint_path)

        self.assertEqual(result.name, "Ours")
        self.assertAlmostEqual(result.tardiness_rate, 0.6403940886699507)
        self.assertEqual(result.makespan, 3055)


if __name__ == "__main__":
    unittest.main()
