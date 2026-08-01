# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Server-side speculative-training capture sink.

Under ``--enable-spec-capture``, a request's ``spec_capture`` dictionary asks
the server to write captured prefill tensors directly into Mooncake using the
SpecForge DataFlow key layout. Tensor bytes never travel through the HTTP
response; the response contains only keys, shapes, and dtypes.

Request schema::

    {"store_id", "sample_id", "gen", "replace",
     "features": {"aux": <name>, "last_hidden": <name>},
     "passthrough": [{"name", "data", "shape", "dtype"}]}
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DTYPE_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_DTYPE = {value: key for key, value in _DTYPE_STR.items()}

_ARTIFACT_AUX = "aux"
_ARTIFACT_LAST_HIDDEN = "last_hidden"


class SpecCaptureSink:
    """Write captured per-request tensors into Mooncake."""

    def __init__(self, aux_layer_ids: list[int] | None = None) -> None:
        self.aux_layer_ids = list(aux_layer_ids) if aux_layer_ids else None
        self._store = None
        self._put_config = None
        self._lock = threading.Lock()
        # Retried requests reuse deterministic keys. Striped locks keep a
        # replacement atomic without retaining one lock per sample.
        self._write_locks = [threading.Lock() for _ in range(256)]

    def _connect(self):
        if self._store is not None:
            return self._store
        with self._lock:
            if self._store is not None:
                return self._store
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig

            store = MooncakeDistributedStore()
            status = store.setup(
                local_hostname=os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
                metadata_server=os.environ.get(
                    "MOONCAKE_METADATA_SERVER",
                    "http://localhost:8080/metadata",
                ),
                global_segment_size=int(
                    os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", "1073741824")
                ),
                local_buffer_size=int(
                    os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", "1073741824")
                ),
                protocol=os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
                rdma_devices=os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
                master_server_addr=os.environ.get(
                    "MOONCAKE_MASTER_SERVER_ADDR", "localhost:50051"
                ),
            )
            if status is not None and int(status) != 0:
                raise RuntimeError(
                    f"spec-capture Mooncake setup failed (status {status})"
                )
            config = ReplicateConfig()
            config.replica_num = 1
            # SpecForge, rather than Mooncake's LRU, owns feature lifetime.
            config.with_hard_pin = True
            self._put_config = config
            self._store = store
            logger.info("spec-capture Mooncake sink connected")
            return store

    @staticmethod
    def _tensor_key(store_id: str, sample_id: str, gen: int, name: str) -> str:
        return f"{store_id}/{sample_id}/g{gen}/{name}"

    def _put_tensor(
        self, key: str, tensor: torch.Tensor, *, replace: bool = False
    ) -> None:
        store = self._connect()
        tensor = tensor.detach().to("cpu").contiguous()
        nbytes = tensor.element_size() * tensor.numel()
        lock = self._write_locks[hash(key) % len(self._write_locks)]
        with lock:
            if replace:
                # Avoid is_exist(): some Mooncake builds acquire a read lease
                # that prevents the following removal.
                self._remove_quiet(key)
            try:
                store.register_buffer(tensor.data_ptr(), nbytes)
            except Exception as error:  # noqa: BLE001
                logger.debug("Mooncake buffer registration was skipped: %s", error)
            try:
                status = store.put_from(
                    key,
                    tensor.data_ptr(),
                    nbytes,
                    self._put_config,
                )
            finally:
                try:
                    store.unregister_buffer(tensor.data_ptr())
                except Exception as error:  # noqa: BLE001
                    logger.debug(
                        "Mooncake buffer unregistration was skipped: %s", error
                    )
        if status is not None and int(status) < 0:
            raise RuntimeError(
                f"spec-capture put_from failed (status {status}) for {key}"
            )

    def _remove_quiet(self, key: str) -> None:
        try:
            self._connect().remove(key)
        except Exception as error:  # noqa: BLE001
            logger.debug("Mooncake key removal was skipped for %s: %s", key, error)

    def put_sample(
        self,
        spec: dict[str, Any],
        *,
        aux: torch.Tensor | None,
        last_hidden: torch.Tensor | None,
    ) -> dict[str, Any]:
        """Write one sample and return its response metadata."""

        store_id = str(spec["store_id"])
        sample_id = str(spec["sample_id"])
        gen = int(spec.get("gen", 1))
        replace = bool(spec.get("replace", False))
        features: dict[str, str] = dict(spec.get("features") or {})
        written: list[str] = []
        result_features: dict[str, dict[str, Any]] = {}

        def _write(name: str, tensor: torch.Tensor) -> None:
            key = self._tensor_key(store_id, sample_id, gen, name)
            self._put_tensor(key, tensor, replace=replace)
            written.append(key)
            result_features[name] = {
                "shape": list(tensor.shape),
                "dtype": _DTYPE_STR.get(
                    tensor.dtype, str(tensor.dtype).replace("torch.", "")
                ),
            }

        try:
            aux_name = features.get(_ARTIFACT_AUX)
            if aux_name is not None:
                if aux is None:
                    raise RuntimeError(
                        "spec_capture requested aux but no auxiliary hidden "
                        "states were captured"
                    )
                _write(aux_name, aux.unsqueeze(0))
            last_hidden_name = features.get(_ARTIFACT_LAST_HIDDEN)
            if last_hidden_name is not None:
                if last_hidden is None:
                    raise RuntimeError(
                        "spec_capture requested last_hidden but it was not captured"
                    )
                _write(last_hidden_name, last_hidden.unsqueeze(0))
            for item in spec.get("passthrough") or []:
                dtype = _STR_DTYPE.get(str(item.get("dtype", "int64")))
                if dtype is None:
                    raise RuntimeError(
                        f"spec_capture passthrough {item.get('name')!r}: "
                        f"unsupported dtype {item.get('dtype')!r}"
                    )
                tensor = torch.tensor(item["data"], dtype=dtype).reshape(
                    [int(dimension) for dimension in item["shape"]]
                )
                _write(str(item["name"]), tensor)
        except Exception:
            for key in written:
                self._remove_quiet(key)
            raise

        return {
            "sample_id": sample_id,
            "store_id": store_id,
            "gen": gen,
            "aux_layer_ids": self.aux_layer_ids,
            "features": result_features,
        }


_SINK: SpecCaptureSink | None = None


def maybe_init_sink(server_args) -> None:
    """Initialize the process-local sink; Mooncake connection remains lazy."""

    global _SINK
    if getattr(server_args, "enable_spec_capture", False) and _SINK is None:
        _SINK = SpecCaptureSink(
            aux_layer_ids=getattr(server_args, "spec_capture_aux_layer_ids", None)
        )


def get_sink() -> SpecCaptureSink | None:
    return _SINK
