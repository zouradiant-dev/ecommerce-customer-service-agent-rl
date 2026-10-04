"""Deployment must preserve unknown changes and keep exact recoverable originals."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reward_installer", ROOT / "scripts/install_reward_fix_v12.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo, self.package = self.root / "repo", self.root / "package"
        (self.repo / "src/ecommerce_customer_service").mkdir(parents=True)
        self.tools = self.repo / "src/ecommerce_customer_service/tools.py"
        self.tools.write_text("frozen", encoding="utf-8")
        (self.package / "payload/src").mkdir(parents=True)
        self.target = self.repo / "src/reward.py"
        self.target.write_text("old", encoding="utf-8")
        self.payload = self.package / "payload/src/reward.py"
        self.payload.write_text("new", encoding="utf-8")
        self.manifest = {"verifier_version": "deterministic_semantic_v12",
            "frozen_files": {"src/ecommerce_customer_service/tools.py": installer.digest(self.tools)},
            "payload_sha256": {"src/reward.py": installer.digest(self.payload)},
            "previous_sha256": {"src/reward.py": [installer.digest(self.target)]}}
        self.save_manifest()

    def save_manifest(self):
        (self.package / "bundle.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_backup_and_idempotent_install(self):
        result = installer.install(self.package, self.repo)
        self.assertEqual(self.target.read_text(), "new")
        self.assertEqual((Path(result["backup"]) / "src/reward.py").read_text(), "old")
        self.assertFalse(result["gpu_started"])
        self.assertEqual(installer.install(self.package, self.repo)["changed_files"], 0)

    def test_unknown_change_is_not_overwritten(self):
        self.target.write_text("user change", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected_existing_file"):
            installer.install(self.package, self.repo)
        self.assertEqual(self.target.read_text(), "user change")
        self.assertFalse((self.repo / "outputs").exists())

    def test_corrupt_payload_does_not_touch_target(self):
        self.payload.write_text("corrupt", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "payload_modified"):
            installer.install(self.package, self.repo)
        self.assertEqual(self.target.read_text(), "old")

    def test_changed_business_tools_block_entire_installation(self):
        self.tools.write_text("changed tools", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "business_tools_changed"):
            installer.install(self.package, self.repo)
        self.assertEqual(self.target.read_text(), "old")

    def test_unsafe_path_blocks_before_writes(self):
        self.manifest["payload_sha256"]["../outside.py"] = "fake"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "unsafe_bundle_path"):
            installer.install(self.package, self.repo)
        self.assertEqual(self.target.read_text(), "old")


if __name__ == "__main__":
    unittest.main()
