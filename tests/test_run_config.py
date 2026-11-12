import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_config", Path(__file__).resolve().parents[1] / "scripts" / "grasp" / "run_config.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloth_friction", type=float, default=None)
    parser.add_argument("--table_friction", type=float, default=None)
    parser.add_argument("--config")
    return parser


class RunConfigTest(unittest.TestCase):
    def write(self, data):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_repo_config_matches_team_table(self):
        values = MODULE.load_config(MODULE.DEFAULT_CONFIG)
        self.assertEqual((values["cloth_size"], values["cloth_thickness"], values["cloth_density"], values["youngs_modulus"],
                          values["damping"], values["mesh_refinement"]), (0.30, 0.003, 180.0, 1e6, 0.25, 32))
        self.assertNotIn("bend_stiffness", values)  # still being searched (009)
        self.assertFalse(any(key.startswith("_") for key in values))

    def test_file_sets_defaults_and_command_line_wins(self):
        path = self.write({"_note": "skipped", "cloth_friction": 0.5, "table_friction": 0.3})
        parser = make_parser()
        argv = ["--config", path, "--cloth_friction", "0.8"]
        MODULE.apply_config_defaults(parser, argv)
        args = parser.parse_args(argv)
        self.assertEqual((args.cloth_friction, args.table_friction), (0.8, 0.3))

    def test_none_skips_the_file(self):
        parser = make_parser()
        self.assertIsNone(MODULE.apply_config_defaults(parser, ["--config", "none"]))
        self.assertIsNone(parser.parse_args(["--config", "none"]).cloth_friction)

    def test_unknown_key_is_an_error(self):
        path = self.write({"clothfriction": 0.5})
        with self.assertRaises(SystemExit):
            MODULE.apply_config_defaults(make_parser(), ["--config", path])


if __name__ == "__main__":
    unittest.main()
