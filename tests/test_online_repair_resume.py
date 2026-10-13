from pathlib import Path
from hashlib import sha256
import importlib.util
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('resume_repair',ROOT/'scripts/resume_online_repair_evaluation.py')
resume=importlib.util.module_from_spec(spec);spec.loader.exec_module(resume)


class ResumeSafetyTests(unittest.TestCase):
    def test_patch_preserves_weights_baseline_and_original_source_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo=Path(temporary)/'repo';target=repo/'src/ecommerce_customer_service/serving/artifacts.py'
            target.parent.mkdir(parents=True);target.write_bytes(b'old compatibility source')
            weights=repo/'outputs/online-repair-pilot-r1/pilot/adapter_model.safetensors';weights.parent.mkdir(parents=True);weights.write_bytes(b'completed weights')
            baseline=repo/'outputs/online-repair-pilot-r1/baseline/summary.json';baseline.parent.mkdir(parents=True);baseline.write_bytes(b'original measured results')
            patch=Path(temporary)/'patch.py';patch.write_bytes(b'fixed compatibility source')
            receipt=resume.install_patch(repo,patch,expected_old=sha256(b'old compatibility source').hexdigest())
            self.assertFalse(receipt['training_or_weights_modified'])
            self.assertEqual(b'completed weights',weights.read_bytes())
            self.assertEqual(b'original measured results',baseline.read_bytes())
            self.assertEqual(b'old compatibility source',target.with_name('artifacts.pre-tokenizer-format-fix.py').read_bytes())
            self.assertEqual(b'fixed compatibility source',target.read_bytes())
            resume.install_patch(repo,patch) # Already applied is safe and idempotent.

    def test_unrecognized_source_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo=Path(temporary)/'repo';target=repo/'src/ecommerce_customer_service/serving/artifacts.py';target.parent.mkdir(parents=True)
            target.write_bytes(b'separate user edit')
            patch=Path(temporary)/'patch.py';patch.write_bytes(b'fix')
            with self.assertRaisesRegex(ValueError,'no_overwrite'): resume.install_patch(repo,patch)
            self.assertEqual(b'separate user edit',target.read_bytes())

    def test_windows_newlines_are_accepted_but_content_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo=Path(temporary)/'repo';target=repo/'src/ecommerce_customer_service/serving/artifacts.py';target.parent.mkdir(parents=True)
            target.write_bytes(b'known\r\nsource\r\n')
            patch=Path(temporary)/'patch.py';patch.write_bytes(b'fixed\nsource\n')
            expected=sha256(b'known\nsource\n').hexdigest()
            receipt=resume.install_patch(repo,patch,expected_old=expected)
            self.assertEqual(expected,receipt['before_lf_normalized_sha256'])
            self.assertEqual(b'known\r\nsource\r\n',target.with_name('artifacts.pre-tokenizer-format-fix.py').read_bytes())


if __name__=='__main__': unittest.main()
