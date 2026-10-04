"""Server baseline allowlists must be backed by exact verified archive bytes."""
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reward_packager", ROOT / "scripts/package_reward_fix_v12.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class ServerBaselineTest(unittest.TestCase):
    def test_deployment_test_installer_dependency_is_in_payload(self):
        self.assertIn("tests/test_reward_fix_deploy.py", packager.PAYLOAD)
        self.assertIn("scripts/install_reward_fix_v12.py", packager.PAYLOAD)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "baseline.tar"
        self.report = self.root / "verification.json"
        self.relative = "src/reward.py"
        self.name = "ecommerce-customer-service-agent-rl/" + self.relative
        self.raw = b"original\r\n"
        with tarfile.open(self.archive, "w:") as handle:
            member = tarfile.TarInfo(self.name)
            member.size = len(self.raw)
            handle.addfile(member, io.BytesIO(self.raw))
        self.evidence = {
            "experiment_backup_verified": True,
            "all_regular_files_read": True,
            "sha256_actual": packager.digest(self.archive),
            "sha256_expected": packager.digest(self.archive),
            "archive_file_manifest": {self.name: {
                "bytes": len(self.raw), "sha256": sha256(self.raw).hexdigest(),
            }},
        }
        self.save_report()

    def save_report(self):
        self.report.write_text(json.dumps(self.evidence), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_verified_server_files_and_lf_equivalent_are_accepted(self):
        previous = packager.verified_server_baseline(self.archive, self.report, {self.relative, "missing.py"})
        self.assertEqual(previous, {self.relative: sorted({
            sha256(self.raw).hexdigest(), sha256(b"original\n").hexdigest(),
        })})

    def test_unverified_report_is_rejected(self):
        self.evidence["all_regular_files_read"] = False
        self.save_report()
        with self.assertRaisesRegex(ValueError, "server_baseline_not_verified"):
            packager.verified_server_baseline(self.archive, self.report, {self.relative})

    def test_changed_archive_is_rejected(self):
        with self.archive.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "server_baseline_archive_hash_mismatch"):
            packager.verified_server_baseline(self.archive, self.report, {self.relative})

    def test_wrong_recorded_file_hash_is_rejected(self):
        self.evidence["archive_file_manifest"][self.name]["sha256"] = "0" * 64
        self.save_report()
        with self.assertRaisesRegex(ValueError, "server_baseline_file_hash_mismatch"):
            packager.verified_server_baseline(self.archive, self.report, {self.relative})


if __name__ == "__main__":
    unittest.main()
