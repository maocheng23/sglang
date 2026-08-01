import unittest

import torch

from sglang.srt.spec_capture_sink import SpecCaptureSink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeStore:
    def __init__(self, *, fail_at=None):
        self.fail_at = fail_at
        self.puts = []
        self.removed = []

    def register_buffer(self, pointer, nbytes):
        return None

    def unregister_buffer(self, pointer):
        return None

    def put_from(self, key, pointer, nbytes, config):
        if self.fail_at is not None and len(self.puts) == self.fail_at:
            raise RuntimeError("injected put failure")
        self.puts.append((key, nbytes))
        return 0

    def remove(self, key):
        self.removed.append(key)


class TestSpecCaptureSink(unittest.TestCase):
    def _sink(self, store):
        sink = SpecCaptureSink(aux_layer_ids=[11, 23, 47, 71, 83])
        sink._store = store
        sink._put_config = object()
        return sink

    def test_writes_features_and_passthrough_in_specforge_layout(self):
        store = _FakeStore()
        sink = self._sink(store)
        result = sink.put_sample(
            {
                "store_id": "run",
                "sample_id": "sample-7",
                "gen": 2,
                "features": {
                    "aux": "hidden_states",
                    "last_hidden": "target_last_hidden_states",
                },
                "passthrough": [
                    {
                        "name": "input_ids",
                        "data": [[1, 2, 3]],
                        "shape": [1, 3],
                        "dtype": "int64",
                    }
                ],
            },
            aux=torch.zeros(3, 10, dtype=torch.bfloat16),
            last_hidden=torch.zeros(3, 2, dtype=torch.bfloat16),
        )

        self.assertEqual(
            [key for key, _ in store.puts],
            [
                "run/sample-7/g2/hidden_states",
                "run/sample-7/g2/target_last_hidden_states",
                "run/sample-7/g2/input_ids",
            ],
        )
        self.assertEqual(result["sample_id"], "sample-7")
        self.assertEqual(result["aux_layer_ids"], [11, 23, 47, 71, 83])
        self.assertEqual(
            result["features"]["hidden_states"],
            {"shape": [1, 3, 10], "dtype": "bfloat16"},
        )

    def test_partial_sample_is_removed_after_put_failure(self):
        store = _FakeStore(fail_at=1)
        sink = self._sink(store)
        with self.assertRaisesRegex(RuntimeError, "injected put failure"):
            sink.put_sample(
                {
                    "store_id": "run",
                    "sample_id": "sample-9",
                    "features": {
                        "aux": "hidden_states",
                        "last_hidden": "target_last_hidden_states",
                    },
                },
                aux=torch.zeros(2, 4),
                last_hidden=torch.zeros(2, 4),
            )
        self.assertEqual(store.removed, ["run/sample-9/g1/hidden_states"])


if __name__ == "__main__":
    unittest.main()
