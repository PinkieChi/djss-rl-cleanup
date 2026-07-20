import unittest
from pathlib import Path
import tempfile

from djss_rl.datasets import DatasetSpec, generate_dataset, generate_dataset_from_jsplib, parse_jsplib
from djss_rl.environment import make_env
from djss_rl.evaluation import evaluate_checkpoint, run_scheduling
from djss_rl.experiments import (
    run_baseline_grid,
    run_checkpoint_generalization_study,
    run_paper_study,
    run_policy_trace_study,
    run_rl_generalization_study,
)
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


class ExperimentInfrastructureTest(unittest.TestCase):
    def test_generated_stable_id_dataset_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "generated.ini"
            generate_dataset(
                dataset_path,
                spec=DatasetSpec(
                    jobs=6,
                    work_centers=2,
                    machines_per_work_center=2,
                    min_operations=2,
                    max_operations=3,
                    min_processing_time=10,
                    max_processing_time=20,
                ),
                seed=11,
            )

            env = make_env(dataset_path=dataset_path)

        self.assertEqual(len(env.world.jobs), 6)
        self.assertEqual(len(env.world.machines), 4)
        self.assertEqual(env.observation_space.shape, (14,))

    def test_jsplib_conversion_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "tiny_jsplib.txt"
            source_path.write_text(
                "instance tiny\n# jobs machines\n2 2\n0 3 1 2\n1 2 0 4\n",
                encoding="utf-8",
            )
            output_path = Path(tmpdir) / "tiny_jsplib.ini"

            jobs = parse_jsplib(source_path)
            generate_dataset_from_jsplib(source_path, output_path, initial_job_fraction=1.0)
            env = make_env(dataset_path=output_path)

        self.assertEqual(jobs, [[(1, 3), (2, 2)], [(2, 2), (1, 4)]])
        self.assertEqual(len(env.world.jobs), 2)
        self.assertEqual(len(env.world.machines), 2)
        self.assertEqual(env.world.operations, 4)

    def test_tiny_baseline_grid_writes_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, summary_path = run_baseline_grid(
                output_dir=tmpdir,
                jobs_values=[6],
                ddt_values=[0.5],
                arrival_rates=[50],
                seeds=[11],
                work_centers=2,
                machines_per_work_center=2,
                min_operations=2,
                max_operations=3,
                min_processing_time=10,
                max_processing_time=20,
            )

            csv_text = csv_path.read_text(encoding="utf-8")
            summary_text = summary_path.read_text(encoding="utf-8")

        self.assertIn("MRT_DR_O", csv_text)
        self.assertIn("SPT_DR_O", csv_text)
        self.assertIn("Experiment Matrix Summary", summary_text)
        self.assertIn("Median", summary_text)
        self.assertIn("Rank-biserial", summary_text)
        self.assertIn("Regime Breakdown", summary_text)

    def test_tiny_rl_study_writes_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, summary_path = run_rl_generalization_study(
                output_dir=tmpdir,
                jobs_values=[6],
                ddt_values=[0.5],
                arrival_rates=[50],
                train_instance_seeds=[11],
                validation_instance_seeds=[44],
                test_instance_seeds=[22],
                training_seeds=[33],
                episodes=1,
                validation_every=1,
                work_centers=2,
                machines_per_work_center=2,
                min_operations=2,
                max_operations=3,
                min_processing_time=10,
                max_processing_time=20,
            )

            csv_text = csv_path.read_text(encoding="utf-8")
            summary_text = summary_path.read_text(encoding="utf-8")
            config_exists = (Path(tmpdir) / "study_config.json").exists()

        self.assertIn("DQN", csv_text)
        self.assertIn("SPT_DR_O", csv_text)
        self.assertIn("validation", csv_text)
        self.assertIn("RL Generalization Study Summary", summary_text)
        self.assertIn("DQN Against Baselines", summary_text)
        self.assertIn("Median", summary_text)
        self.assertIn("Regime Breakdown", summary_text)
        self.assertTrue(config_exists)

    def test_tiny_paper_study_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, summary_path = run_paper_study(
                output_dir=tmpdir,
                variants=["dense"],
                jobs_values=[6],
                ddt_values=[0.5],
                arrival_rates=[50],
                train_instance_seeds=[11],
                validation_instance_seeds=[44],
                test_instance_seeds=[22],
                training_seeds=[33],
                episodes=1,
                validation_every=1,
                work_centers=2,
                machines_per_work_center=2,
                min_operations=2,
                max_operations=3,
                min_processing_time=10,
                max_processing_time=20,
            )

            csv_text = csv_path.read_text(encoding="utf-8")
            summary_text = summary_path.read_text(encoding="utf-8")

        self.assertIn("dense", csv_text)
        self.assertIn("Paper Study Summary", summary_text)
        self.assertIn("DQN-SPT", summary_text)

    def test_tiny_checkpoint_study_writes_summary(self):
        project_dir = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            project_dir
            / "Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, summary_path = run_checkpoint_generalization_study(
                output_dir=tmpdir,
                checkpoint_paths=[checkpoint_path],
                checkpoint_labels=["restored"],
                jobs_values=[6],
                ddt_values=[0.5],
                arrival_rates=[50],
                test_instance_seeds=[22],
                work_centers=2,
                machines_per_work_center=2,
                min_operations=2,
                max_operations=3,
                min_processing_time=10,
                max_processing_time=20,
            )

            csv_text = csv_path.read_text(encoding="utf-8")
            summary_text = summary_path.read_text(encoding="utf-8")

        self.assertIn("DQN", csv_text)
        self.assertIn("restored", csv_text)
        self.assertIn("RL Generalization Study Summary", summary_text)
        self.assertIn("Rank-biserial", summary_text)
        self.assertIn("Publication Use", summary_text)

    def test_tiny_policy_trace_writes_summary(self):
        project_dir = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            project_dir
            / "Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "generated.ini"
            generate_dataset(
                dataset_path,
                spec=DatasetSpec(
                    jobs=6,
                    work_centers=2,
                    machines_per_work_center=2,
                    min_operations=2,
                    max_operations=3,
                    min_processing_time=10,
                    max_processing_time=20,
                ),
                seed=11,
            )
            csv_path, summary_path = run_policy_trace_study(
                output_dir=Path(tmpdir) / "trace",
                checkpoint_path=checkpoint_path,
                dataset_paths=[dataset_path],
            )

            csv_text = csv_path.read_text(encoding="utf-8")
            summary_text = summary_path.read_text(encoding="utf-8")

        self.assertIn("dominant_action", csv_text)
        self.assertIn("Policy Trace Summary", summary_text)
        self.assertIn("Action Distribution", summary_text)


if __name__ == "__main__":
    unittest.main()
