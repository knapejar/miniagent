# -*- coding: utf-8 -*-
"""Profile selection: the wafer backend, listing, and the stored default."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config, paths


class TestProfiles(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.saved = {k: os.environ.get(k) for k in (paths.ENV_HOME, "MINIAGENT_PROFILE")}
        os.environ[paths.ENV_HOME] = self.home
        os.environ.pop("MINIAGENT_PROFILE", None)

    def tearDown(self):
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_wafer_profile(self):
        cfg, _ = config.parse_args(["--profile", "wafer"])
        self.assertEqual((cfg.profile, cfg.model, cfg.effort),
                         ("wafer", "GLM-5.3", "low"))
        self.assertTrue(cfg.url.startswith("https://pass.wafer.ai"))

    def test_profile_without_a_name_lists_them(self):
        with self.assertRaises(SystemExit) as caught:
            config.parse_args(["--profile"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("wafer", config.profiles_text())

    def test_default_profile_is_stored_and_used(self):
        self.assertEqual(config.default_profile(), "local")
        config.set_default_profile("wafer")
        self.assertEqual(config.default_profile(), "wafer")
        self.assertEqual(config.parse_args([])[0].profile, "wafer")
        os.environ["MINIAGENT_PROFILE"] = "wingpu"      # the env wins
        self.assertEqual(config.default_profile(), "wingpu")

    def test_unknown_profile_is_refused(self):
        with self.assertRaises(SystemExit):
            config.parse_args(["--profile", "nope"])


if __name__ == "__main__":
    unittest.main()
