import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_grpo_smoke_result import tensor_comparison


class TensorAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def make(self, name, dtype, payload, shape):
        path = self.root / name
        header = json.dumps({'tensor': {'dtype': dtype, 'shape': shape,
                                     'data_offsets': [0, len(payload)]}}).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header + payload)
        return path

    def test_bf16_cast_is_not_counted_as_training_update(self):
        left = self.make('a', 'F32', struct.pack('<fff', 1., 1.00390625, 1.01171875), [3])
        right = self.make('b', 'BF16', struct.pack('<HHH', 0x3f80, 0x3f80, 0x3f82), [3])
        result = tensor_comparison(left, right)
        self.assertEqual(result['changed_tensors'], 0)
        self.assertTrue(result['comparison_after_dtype_alignment_equal'])
        self.assertFalse(result['exact_payload_equal'])

    def test_actual_update_after_cast_is_detected(self):
        left = self.make('a', 'F32', struct.pack('<f', 1.), [1])
        right = self.make('b', 'BF16', struct.pack('<H', 0x3f81), [1])
        self.assertEqual(tensor_comparison(left, right)['changed_tensors'], 1)

    def test_same_dtype_exact_payloads_are_verified(self):
        left = self.make('a', 'BF16', struct.pack('<H', 0x3f80), [1])
        right = self.make('b', 'BF16', struct.pack('<H', 0x3f80), [1])
        self.assertTrue(tensor_comparison(left, right)['exact_payload_equal'])

    def test_truncated_payload_is_rejected(self):
        left = self.make('a', 'F32', struct.pack('<f', 1.), [1])
        right = self.make('b', 'BF16', struct.pack('<H', 0x3f80), [1])
        right.write_bytes(right.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'truncated'):
            tensor_comparison(left, right)


if __name__ == '__main__': unittest.main()
