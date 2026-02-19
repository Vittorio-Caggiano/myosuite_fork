from __future__ import annotations

# Source: https://github.dev/aravindr93/mjrl/tree/master/mjrl
import operator
from typing import Any

import numpy as np


def flatten_tensors(tensors: list[np.ndarray]) -> np.ndarray:
    if len(tensors) > 0:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    else:
        return np.asarray([])


def unflatten_tensors(flattened: np.ndarray, tensor_shapes: list[tuple[int, ...]]) -> list[np.ndarray]:
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [np.reshape(pair[0], pair[1]) for pair in zip(np.split(flattened, indices), tensor_shapes)]


def pad_tensor(x: np.ndarray, max_len: int, mode: str = 'zero') -> np.ndarray:
    padding = np.zeros_like(x[0])
    if mode == 'last':
        padding = x[-1]
    return np.concatenate([
        x,
        np.tile(padding, (max_len - len(x),) + (1,) * np.ndim(x[0]))
    ])


def pad_tensor_n(xs: np.ndarray, max_len: int) -> np.ndarray:
    ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
    for idx, x in enumerate(xs):
        ret[idx][:len(x)] = x
    return ret


def pad_tensor_dict(tensor_dict: dict[str, Any], max_len: int, mode: str = 'zero') -> dict[str, Any]:
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len, mode=mode)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len, mode=mode)
    return ret


def flatten_first_axis_tensor_dict(tensor_dict: dict[str, Any]) -> dict[str, Any]:
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = flatten_first_axis_tensor_dict(tensor_dict[k])
        else:
            old_shape = tensor_dict[k].shape
            ret[k] = tensor_dict[k].reshape((-1,) + old_shape[2:])
    return ret


def high_res_normalize(probs: list[float]) -> list[float]:
    return [x / sum(map(float, probs)) for x in list(map(float, probs))]


def stack_tensor_list(tensor_list: list[np.ndarray]) -> np.ndarray:
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)


def stack_tensor_dict_list(tensor_dict_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def concat_tensor_list_subsample(tensor_list: list[np.ndarray], f: float) -> np.ndarray:
    return np.concatenate(
        [t[np.random.choice(len(t), int(np.ceil(len(t) * f)), replace=False)] for t in tensor_list], axis=0)


def concat_tensor_dict_list_subsample(tensor_dict_list: list[dict[str, Any]], f: float) -> dict[str, Any]:
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list_subsample([x[k] for x in tensor_dict_list], f)
        else:
            v = concat_tensor_list_subsample([x[k] for x in tensor_dict_list], f)
        ret[k] = v
    return ret


def concat_tensor_list(tensor_list: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list: list[dict[str, Any]]) -> dict[str, Any]:
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def split_tensor_dict_list(tensor_dict: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(tensor_dict.keys())
    ret = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret


def truncate_tensor_list(tensor_list: list[np.ndarray], truncated_len: int) -> list[np.ndarray]:
    return tensor_list[:truncated_len]


def truncate_tensor_dict(tensor_dict: dict[str, Any], truncated_len: int) -> dict[str, Any]:
    ret = dict()
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            ret[k] = truncate_tensor_dict(v, truncated_len)
        else:
            ret[k] = truncate_tensor_list(v, truncated_len)
    return ret
