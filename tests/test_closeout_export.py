from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import tempfile
import tarfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('closeout_export',ROOT/'scripts/export_current_instance_closeout.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CloseoutExportTests(unittest.TestCase):
    def fixture(self,work):
        root=work/'current-instance-closeout-20261014.fixture';root.mkdir()
        (work/'current-instance-closeout-20261014-r1.path').write_text(str(root),encoding='utf-8')
        out=root/'outputs/closeout'
        def write(name,value):
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
        write('completed.json',{'complete':True})
        write('sealed-test/summary.json',{'complete':True,'passed_cases':10,'evaluated_cases':30})
        write('http-integration/report.json',{'passed':True})
        write('selection.json',{'selected':'pilot','selected_adapter_sha256':sha256(b'pilot').hexdigest()})
        write('sealed-test-claim.json',{'claimed':True})
        write('recovery-preflight.json',{'fixture':True})
        for phase in ('smoke','pilot'):
            path=out/phase/'adapter_model.safetensors';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(phase.encode())
        write('pilot/current-instance-artifact.json',{'files_sha256':{'adapter_model.safetensors':sha256(b'pilot').hexdigest()}})
        (out/'pilot/training_state.pt').write_bytes(b'optimizer-not-needed')
        checkpoint=out/'pilot/checkpoint-32';checkpoint.mkdir();(checkpoint/'adapter_model.safetensors').write_bytes(b'checkpoint')
        live=out/'selected-service-diagnostics';live.mkdir();(live/'live.json').write_text('{}')
        return root

    def test_final_adapters_and_poor_test_result_preserved_without_optimizer_or_live_files(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);self.fixture(work)
            report=module.export(work)
            with tarfile.open(report['archive']) as archive:
                names=set(archive.getnames())
                self.assertIn('outputs/closeout/pilot/adapter_model.safetensors',names)
                self.assertIn('outputs/closeout/smoke/adapter_model.safetensors',names)
                self.assertFalse(any('checkpoint-' in name or name.endswith('.pt') or 'selected-service-diagnostics' in name for name in names))
                receipt=json.load(archive.extractfile('export-receipt.json'))
                self.assertEqual(10,receipt['sealed_test_passed'])
                for name,digest in receipt['files_sha256'].items():
                    self.assertEqual(digest,sha256(archive.extractfile(name).read()).hexdigest())
            self.assertTrue(module.export(work)['reused'])

    def test_changed_weights_or_unfinished_run_cannot_export(self):
        for broken in ('weights','complete'):
            with self.subTest(broken=broken),tempfile.TemporaryDirectory() as temp:
                work=Path(temp);root=self.fixture(work)
                if broken=='weights':(root/'outputs/closeout/pilot/adapter_model.safetensors').write_bytes(b'changed')
                else:(root/'outputs/closeout/completed.json').write_text('{"complete":false}')
                with self.assertRaises(ValueError): module.export(work)
                self.assertFalse((work/'current-instance-closeout-20261014-r1-results.tar.gz').exists())


if __name__=='__main__': unittest.main()
