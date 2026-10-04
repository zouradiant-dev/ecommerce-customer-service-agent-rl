"""Synthetic, isolated packaging/install tests. No real review or GPU results."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import install_grpo_experiment_v13 as installer
import package_grpo_experiment_v13 as packager


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline"
        self.evidence = self.root / "synthetic-evidence"
        self.review = self.root / "synthetic-review"
        self.repo = self.root / "synthetic-repo"
        self.evidence.mkdir()
        self.review.mkdir()
        hashes = {}
        for name in ("src/ecommerce_customer_service/online_env.py", "src/ecommerce_customer_service/simulator.py"):
            path = self.baseline / "payload" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# synthetic previous version, not executable evidence\n")
            hashes[name] = packager.digest(path)
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (self.baseline / "bundle.json").write_text(json.dumps({"verifier_version": "deterministic_semantic_v12", "payload_sha256": hashes}))
        (self.evidence / "protocol.json").write_text(json.dumps({"generation_source_sha256": {
            "src/ecommerce_customer_service/simulator.py": hashes["src/ecommerce_customer_service/simulator.py"]}}))
        rows = self.evidence / "sft/rollouts.jsonl"
        rows.parent.mkdir()
        rows.write_text(json.dumps({"task_id": "fixture", "trial": 0, "reward": 0.0}) + "\n")
        (self.review / "review.json").write_text(json.dumps({"human_signoff": True,
            "reviewer": "synthetic fixture; no real human approval", "reviewed_at_utc": "2026-10-02T00:00:00+00:00",
            "verifier_version": "deterministic_semantic_v13", "rollouts_sha256": packager.digest(rows),
            "rows": [{"task_id": "fixture", "trial": 0, "recorded_reward": 0.0, "expected_reward": 0.0,
                      "status": "confirmed", "evidence": "synthetic fixture, not real readiness"}]}))
        self.package = self.root / "test-package"
        self.report = packager.build(self.baseline, self.evidence, self.review, self.package)
        self.manifest = json.loads((self.package / "bundle.json").read_text(encoding="utf-8"))
        for name in self.manifest["frozen_files"]:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.sentinel = self.repo / "outputs/original-report.json"
        self.sentinel.parent.mkdir()
        self.sentinel.write_bytes(b"user original result must survive\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_archive_is_verified_checksum_is_linux_lf_and_no_weights(self):
        archive = Path(self.report["archive"])
        checksum = archive.with_name(archive.name + ".sha256").read_bytes()
        self.assertNotIn(b"\r", checksum)
        self.assertEqual(checksum.decode().split()[0], packager.digest(archive))
        self.assertTrue(self.report["archive_payload_verified"])
        self.assertFalse(self.report["gpu_started"])
        self.assertFalse(self.manifest["weights_included"])
        self.assertFalse(self.manifest["formal_rl_gate_passed"])

    def test_installed_test_helpers_import_in_fresh_isolated_interpreter(self):
        helpers = {"scripts/install_grpo_experiment_v13.py", "scripts/package_grpo_experiment_v13.py",
                   "scripts/install_reward_fix_v12.py"}
        self.assertTrue(helpers.issubset(self.manifest["payload_sha256"]))
        installer.install(self.package, self.repo)
        # Supply unchanged application modules, never the original scripts path.
        for source in (ROOT / "src").rglob("*.py"):
            target = self.repo / "src" / source.relative_to(ROOT / "src")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        code = ("from pathlib import Path; import sys; root=Path(sys.argv[1]).resolve(); "
                "sys.path[:0]=[str(root/'scripts'),str(root/'src')]; "
                "import install_grpo_experiment_v13 as a, package_grpo_experiment_v13 as b, install_reward_fix_v12 as c; "
                "assert all(Path(m.__file__).resolve().is_relative_to(root/'scripts') for m in (a,b,c)); "
                "assert b.ROOT==root")
        result = subprocess.run([sys.executable, "-I", "-X", "utf8=0", "-c", code, str(self.repo)],
                                cwd=self.repo, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_verified_previous_release_upgrade_preserves_unknown_edit_guard(self):
        upgrade = self.root / "upgrade-package"
        packager.build(self.baseline, self.evidence, self.review, upgrade, previous_release=self.package)
        manifest = json.loads((upgrade / "bundle.json").read_text(encoding="utf-8"))
        for name, expected in self.manifest["payload_sha256"].items():
            self.assertIn(expected, manifest["previous_sha256"][name])
        installer.install(self.package, self.repo)
        target = self.repo / "scripts/package_grpo_experiment_v13.py"
        target.write_bytes(b"unknown user modification\n")
        with self.assertRaisesRegex(ValueError, "unexpected_existing_file"):
            installer.install(upgrade, self.repo)
        self.assertEqual(target.read_bytes(), b"unknown user modification\n")
        (self.package / "payload/docs/grpo-experiment-v13.md").write_bytes(b"tampered previous release\n")
        with self.assertRaisesRegex(ValueError, "previous_release_modified"):
            packager.build(self.baseline, self.evidence, self.review, self.root / "blocked-package",
                           previous_release=self.package)
        self.assertFalse((self.root / "blocked-package").exists())

    def test_install_backup_preserves_legacy_and_original_reports(self):
        path = "src/ecommerce_customer_service/online_env.py"
        original = (self.repo / path).read_bytes()
        result = installer.install(self.package, self.repo)
        self.assertEqual((Path(result["backup"]) / path).read_bytes(), original)
        self.assertEqual(self.sentinel.read_bytes(), b"user original result must survive\n")
        self.assertEqual(result["legacy_default_version"], "deterministic_semantic_v12")
        self.assertFalse(result["gpu_started"])
        self.assertEqual(installer.install(self.package, self.repo)["changed_files"], 0)

    def test_unknown_user_change_blocks_all_target_writes(self):
        target = self.repo / "src/ecommerce_customer_service/online_env.py"
        target.write_bytes(b"unrecognized user edit\n")
        with self.assertRaisesRegex(ValueError, "unexpected_existing_file"):
            installer.install(self.package, self.repo)
        self.assertEqual(target.read_bytes(), b"unrecognized user edit\n")
        self.assertFalse((self.repo / "scripts/train_grpo_experiment_v13.py").exists())

    def test_tampered_review_or_evidence_cannot_be_installed(self):
        path = self.package / "review-confirmed/review.json"
        saved = path.read_bytes()
        path.write_bytes(b"tampered review\n")
        with self.assertRaisesRegex(ValueError, "human_review_modified"):
            installer.install(self.package, self.repo)
        path.write_bytes(saved)
        (self.package / "evidence/sft/rollouts.jsonl").write_bytes(b"tampered trajectory\n")
        with self.assertRaisesRegex(ValueError, "evidence_modified"):
            installer.install(self.package, self.repo)

    def test_tampered_payload_bootstrap_missing_hashes_and_scope_are_rejected(self):
        common = self.package / "install_reward_fix_v12.py"
        saved = common.read_bytes()
        common.write_bytes(b"tampered installer\n")
        with self.assertRaisesRegex(ValueError, "bootstrap_modified"):
            installer.install(self.package, self.repo)
        common.write_bytes(saved)
        manifest_path = self.package / "bundle.json"
        changed = dict(self.manifest)
        changed["bootstrap_sha256"] = {}
        manifest_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "missing_bootstrap"):
            installer.install(self.package, self.repo)
        changed = dict(self.manifest)
        changed["formal_rl_gate_passed"] = True
        manifest_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "bundle_contract"):
            installer.install(self.package, self.repo)
        manifest_path.write_text(json.dumps(self.manifest))
        (self.package / "payload/scripts/train_grpo_experiment_v13.py").write_bytes(b"tampered trainer\n")
        with self.assertRaisesRegex(ValueError, "payload_modified"):
            installer.install(self.package, self.repo)


if __name__ == "__main__":
    unittest.main()
