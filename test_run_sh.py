import os
import stat
import unittest


SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "run.sh")


class TestRunSh(unittest.TestCase):

    def test_script_exists(self):
        self.assertTrue(os.path.isfile(SCRIPT_PATH))

    def test_script_is_executable(self):
        mode = os.stat(SCRIPT_PATH).st_mode
        self.assertTrue(mode & stat.S_IXUSR)

    def test_script_has_shebang(self):
        with open(SCRIPT_PATH) as f:
            first_line = f.readline()
        self.assertTrue(first_line.startswith("#!/bin/bash"))

    def test_script_creates_venv(self):
        with open(SCRIPT_PATH) as f:
            content = f.read()
        self.assertIn("python3 -m venv .venv", content)

    def test_script_activates_venv(self):
        with open(SCRIPT_PATH) as f:
            content = f.read()
        self.assertIn("source .venv/bin/activate", content)

    def test_script_installs_requirements(self):
        with open(SCRIPT_PATH) as f:
            content = f.read()
        self.assertIn("pip install -r requirements.txt", content)

    def test_script_does_not_force_cpu_wheels(self):
        with open(SCRIPT_PATH) as f:
            content = f.read()
        self.assertNotIn("download.pytorch.org/whl/cpu", content)

    def test_script_runs_main(self):
        with open(SCRIPT_PATH) as f:
            content = f.read()
        self.assertIn("python main.py", content)


if __name__ == "__main__":
    unittest.main()
