"""CPU-only guard regressions; actual CUDA/model execution stays server-only."""
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_grpo_smoke_v12_r1 as smoke


class ScalarResult:
    def __init__(self, value): self.value = value
    def all(self): return self
    def item(self): return self.value


class FakeTensor:
    def __init__(self, values): self.values = list(values)
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return FakeTensor(self.values)


class FakeParameter(FakeTensor):
    def __init__(self, values, *, trainable=True):
        super().__init__(values)
        self.requires_grad, self.grad = trainable, None


class FakeTorch:
    @staticmethod
    def isfinite(tensor): return ScalarResult(all(math.isfinite(v) for v in tensor.values))
    @staticmethod
    def count_nonzero(tensor): return ScalarResult(sum(v != 0 for v in tensor.values))
    @staticmethod
    def equal(left, right): return left.values == right.values


class UpdateGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.audit = Path(self.temp.name) / 'gradients.jsonl'
        self.param = FakeParameter([1., 2.])
        self.ref = FakeParameter([1., 2.], trainable=False)
        self.model = SimpleNamespace(named_parameters=lambda: iter([
            ('layer.lora_A.default.weight', self.param), ('layer.lora_A.ref.weight', self.ref)]))
        self.guard = smoke.make_policy_update_guard(object, FakeTorch, self.model, self.audit)
        self.state = SimpleNamespace(global_step=0)

    def tearDown(self): self.temp.cleanup()

    def step(self, gradient):
        self.param.grad = FakeTensor(gradient) if gradient is not None else None
        with redirect_stdout(io.StringIO()):
            self.guard.on_pre_optimizer_step(None, self.state, None)
        self.guard.on_optimizer_step(None, self.state, None)
        self.state.global_step += 1

    def test_real_observation_and_parameter_change_required(self):
        self.step([.1, -.2])
        self.param.values[0] += .001
        result = self.guard.verify(1, 1)
        self.assertTrue(result['actual_policy_update_verified'])
        self.assertEqual(result['changed_policy_tensors'], 1)
        self.assertEqual(self.ref.values, [1., 2.])
        self.assertEqual(json.loads(self.audit.read_text())['tensors_with_nonzero_grad'], 1)

    def test_optimizer_events_without_changed_weights_fail(self):
        self.step([.1, -.2])
        with self.assertRaisesRegex(RuntimeError, 'did_not_change'):
            self.guard.verify(1, 1)

    def test_no_gradients_and_nonfinite_gradients_fail_immediately(self):
        for values, error in ((None, 'gradients_missing'), ([float('nan')], 'nonfinite')):
            with self.subTest(values=values), self.assertRaisesRegex(RuntimeError, error):
                self.step(values)

    def test_all_zero_gradients_cannot_be_reported_as_success(self):
        self.step([0., 0.])
        self.param.values[0] += .001
        with self.assertRaisesRegex(RuntimeError, 'no_nonzero'):
            self.guard.verify(1, 1)

    def test_incomplete_steps_fail(self):
        self.step([.1, -.2])
        self.param.values[0] += .001
        with self.assertRaisesRegex(RuntimeError, 'event_count_mismatch'):
            self.guard.verify(10, 1)

    def test_accidental_reference_unfreeze_fails(self):
        self.ref.requires_grad = True
        with self.assertRaisesRegex(RuntimeError, 'trainable_parameters_changed'):
            self.step([.1, -.2])


if __name__ == '__main__':
    unittest.main()
