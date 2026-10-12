# -*- coding: utf-8 -*-
"""Profiles: the built-in local, your own per-provider JSON files, the stored
default, and starting without a model until /model picks one."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config, connectors, paths                           # noqa: E402
from agent.config import Config                                       # noqa: E402
from agent.llm import LLMClient                                       # noqa: E402
from agent.session import Session                                     # noqa: E402
from agent.tools import default_tools                                 # noqa: E402

# A closed port: nothing answers, and quickly.
HOSTED = {"url": "http://127.0.0.1:9/v1", "api_key_env": "TEST_HOSTED_KEY",
         "headers": {"X-Data-Retention": "none"}, "ctx": 131072, "retries": 2,
         "profiles": {"big": {"model": "some-model-70b"},
                      "small": {"model": "some-model-8b", "effort": "medium"}}}


class TestProfiles(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.saved = {k: os.environ.get(k) for k in
                      (paths.ENV_HOME, "MINIAGENT_PROFILE", "TEST_HOSTED_KEY")}
        os.environ[paths.ENV_HOME] = self.home
        os.environ.pop("MINIAGENT_PROFILE", None)
        os.environ.pop("TEST_HOSTED_KEY", None)
        self.write("hosted.json", HOSTED)

    def tearDown(self):
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def write(self, name, data):
        os.makedirs(os.path.join(self.home, "profiles"), exist_ok=True)
        with open(os.path.join(self.home, "profiles", name), "w", encoding="utf-8") as fh:
            fh.write(data if isinstance(data, str) else json.dumps(data))

    def test_a_provider_file_shares_its_keys(self):
        cfg, _ = config.parse_args(["--profile", "small"])
        self.assertEqual((cfg.profile, cfg.provider, cfg.model, cfg.effort, cfg.ctx, cfg.retries),
                         ("small", "hosted", "some-model-8b", "medium", 131072, 2))
        self.assertEqual(Config(profile="big").effort, "low")     # DEFAULTS fill the rest

    def test_env_and_provider_key_file(self):
        with open(paths.key_file("hosted"), "w") as fh:
            fh.write("from-the-provider-file\n")
        self.assertEqual(Config(profile="small").api_key, "from-the-provider-file")
        os.environ["TEST_HOSTED_KEY"] = "from-the-env"
        self.assertEqual(Config(profile="small").api_key, "from-the-env")

    def test_profile_headers_reach_the_request(self):
        sent = {}
        client = LLMClient(Config(profile="big"))
        client._open = lambda request, cancel=None: sent.update(request.headers) or "ok"
        client._request([], True)
        self.assertEqual(sent.get("X-data-retention"), "none")

    def test_local_is_built_in(self):
        cfg = Config()
        self.assertEqual((cfg.profile, cfg.provider, cfg.model, cfg.connector),
                         ("local", "lmstudio", "spark-x2.5-4b", "lmstudio"))

    def test_interactive_starts_without_a_model(self):
        cfg, task = config.parse_args([])
        self.assertEqual((cfg.profile, cfg.model, task), (None, "", ""))
        cfg, task = config.parse_args(["do it"])          # nobody to ask: local
        self.assertEqual((cfg.profile, task), ("local", "do it"))
        cfg, _ = config.parse_args(["--model", "my-model"])  # no profile file at all
        self.assertEqual((cfg.profile, cfg.model), ("local", "my-model"))
        cfg.use_profile("big")                          # what /model does
        self.assertEqual((cfg.profile, cfg.model), ("big", "some-model-70b"))

    def test_default_profile_is_stored_and_used(self):
        config.set_default_profile("big")
        self.assertEqual(config.parse_args([])[0].profile, "big")
        os.environ["MINIAGENT_PROFILE"] = "small"   # the env wins
        self.assertEqual(config.parse_args([])[0].profile, "small")

    def test_profile_without_a_name_lists_them_with_their_state(self):
        with self.assertRaises(SystemExit) as caught:
            config.parse_args(["--profile"])
        self.assertEqual(caught.exception.code, 0)
        text = connectors.profiles_text()
        self.assertIn("small", text)
        self.assertIn("no API key", text)

    def test_unknown_profile_and_broken_file_are_refused(self):
        with self.assertRaises(SystemExit):
            config.parse_args(["--profile", "nope"])
        self.write("broken.json", "{not json")
        with self.assertRaises(SystemExit):
            config.parse_args(["--profile", "big"])

    def test_llamacpp_status_names_a_missing_build(self):
        self.write("llamacpp.json", {"connector": "llamacpp", "profiles": {"big": {
            "url": "http://127.0.0.1:9/v1", "server": "~/no/such/llama-server.exe",
            "gguf": "~/no/such.gguf"}}})
        ready, note = connectors.status(Config(profile="big"))
        self.assertFalse(ready)
        self.assertIn("llama-server.exe", note)

    def test_switching_model_rewrites_the_tool_format(self):
        cfg = Config(workdir=tempfile.gettempdir())
        session = Session(cfg, default_tools())
        self.assertIn("<tools>", session.messages[0]["content"])        # spark: specs in the prompt
        cfg.use_profile("big")
        session.set_dialect(cfg.dialect)
        self.assertNotIn("<tools>", session.messages[0]["content"])     # native: in the request
        self.assertEqual(session.tool_role, "user")


if __name__ == "__main__":
    unittest.main()
