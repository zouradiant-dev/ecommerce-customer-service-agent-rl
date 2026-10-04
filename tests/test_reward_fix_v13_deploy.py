"""Synthetic deployment fixtures: immutable baselines, backups and no GPU launch."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import install_reward_fix_v13 as installer
import package_reward_fix_v13 as packager


class V13DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v13-bundle-fixture-")
        self.directory = Path(self.temp.name)
        self.baseline = self.directory / "synthetic-v12-baseline"
        self.repo = self.directory / "repo"
        payload_hashes = {}
        for relative in ("src/ecommerce_customer_service/communication.py", "src/ecommerce_customer_service/communication_facts.py",
                         "src/ecommerce_customer_service/online_env.py", "src/ecommerce_customer_service/tools.py"):
            baseline_path = self.baseline / "payload" / relative
            target = self.repo / relative
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, baseline_path)
            shutil.copy2(ROOT / relative, target)
            payload_hashes[relative] = packager.digest(baseline_path)
        for relative in ("src/ecommerce_customer_service/verifier.py", "src/ecommerce_customer_service/rescoring.py"):
            baseline_path = self.baseline / "payload" / relative
            baseline_path.write_bytes(b"# synthetic old code, not runnable training evidence\n")
            shutil.copy2(baseline_path, self.repo / relative)
            payload_hashes[relative] = packager.digest(baseline_path)
        (self.baseline / "bundle.json").write_text(json.dumps({
            "verifier_version": "deterministic_semantic_v12", "payload_sha256": payload_hashes, "frozen_files": {}}), encoding="utf-8")
        self.package = self.directory / "patch"
        self.report = packager.build(self.baseline, self.package)
        self.manifest = json.loads((self.package / "bundle.json").read_text(encoding="utf-8"))
        self.report_sentinel = self.repo / "outputs/original-report.json"
        self.report_sentinel.parent.mkdir()
        self.report_sentinel.write_bytes(b"immutable experiment sentinel\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_verified_archive_has_linux_checksum_and_only_code_targets(self):
        archive = Path(self.report["archive"])
        line = archive.with_name(archive.name + ".sha256").read_bytes()
        self.assertNotIn(b"\r", line)
        self.assertEqual(line.decode().split()[0], packager.digest(archive))
        self.assertEqual(set(self.manifest["payload_sha256"]), installer.PAYLOAD)
        self.assertFalse(any(name.startswith("outputs/") for name in installer.PAYLOAD))
        self.assertTrue(self.report["archive_payload_verified"])
        self.assertFalse(self.report["gpu_started"])

    def test_backup_idempotence_and_reports_preserved(self):
        previous = (self.repo / "src/ecommerce_customer_service/verifier.py").read_bytes()
        result = installer.install(self.package, self.repo)
        self.assertEqual((Path(result["backup"]) / "src/ecommerce_customer_service/verifier.py").read_bytes(), previous)
        self.assertEqual(self.report_sentinel.read_bytes(), b"immutable experiment sentinel\n")
        for relative, digest in self.manifest["payload_sha256"].items():
            self.assertEqual(packager.digest(self.repo / relative), digest)
        self.assertEqual(installer.install(self.package, self.repo)["changed_files"], 0)
        self.assertEqual(result["training_default_version"], "deterministic_semantic_v12")
        self.assertFalse(result["training_entrypoints_changed"])
        self.assertFalse(result["gpu_started"])

    def test_unknown_user_change_blocks_all_payload_writes(self):
        target = self.repo / "src/ecommerce_customer_service/verifier.py"
        target.write_bytes(b"user edit must survive\n")
        with self.assertRaisesRegex(ValueError, "unexpected_existing_file"):
            installer.install(self.package, self.repo)
        self.assertEqual(target.read_bytes(), b"user edit must survive\n")
        self.assertFalse((self.repo / "src/ecommerce_customer_service/communication_facts_v13.py").exists())

    def test_tampered_bootstrap_and_payload_are_rejected(self):
        common = self.package / "install_reward_fix_v12.py"
        original = common.read_bytes()
        common.write_bytes(b"changed installer\n")
        with self.assertRaisesRegex(ValueError, "shared_installer_modified"):
            installer.install(self.package, self.repo)
        common.write_bytes(original)
        (self.package / "payload/src/ecommerce_customer_service/verifier.py").write_bytes(b"corrupt\n")
        with self.assertRaisesRegex(ValueError, "payload_modified"):
            installer.install(self.package, self.repo)
        self.assertFalse((self.repo / "src/ecommerce_customer_service/communication_facts_v13.py").exists())

    def test_default_change_or_extra_target_is_rejected(self):
        manifest = self.package / "bundle.json"
        for mutation in ("default", "target"):
            content = dict(self.manifest)
            content["payload_sha256"] = dict(content["payload_sha256"])
            if mutation == "default":
                content["training_default_version"] = "deterministic_semantic_v13"
            else:
                content["payload_sha256"]["outputs/original-report.json"] = "forged"
            manifest.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaises(ValueError):
                installer.install(self.package, self.repo)
            self.assertEqual(self.report_sentinel.read_bytes(), b"immutable experiment sentinel\n")


if __name__ == "__main__":
    unittest.main()
