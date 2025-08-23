import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
from openpilot.common.swaglog import cloudlog
from openpilot.tools.lib.logreader import LogReader


def deep_flatten_dict(d, sep="/", prefix=None):
  res = {}
  # Flatten dictionaries by iterating keys
  if isinstance(d, dict):
    for key, val in d.items():
      new_prefix = key if prefix is None else f"{prefix}{sep}{key}"
      res.update(deep_flatten_dict(val, sep=sep, prefix=new_prefix))
    return res

  # Flatten ALL lists by iterating indices
  elif isinstance(d, list):
    if not d:
      return {prefix: np.array([])}

    # The special case for scalar lists has been removed.
    # Now, we always expand the list by index.
    for i, item in enumerate(d):
      res.update(deep_flatten_dict(item, sep=sep, prefix=f"{prefix}{sep}{i}"))
    return res

  # Base case: scalar value
  else:
    return {prefix: d}


def get_message_dict(message, typ):
  # This function now expects the specific sub-message, not the whole event.
  if not hasattr(message, 'to_dict') or typ in ('qcomGnss', 'ubloxGnss'):
    return None

  try:
    msg_dict = message.to_dict(verbose=True)
    msg_dict = deep_flatten_dict(msg_dict)
    return msg_dict
  except Exception as e:
    cloudlog.warning(f"Failed to process {typ} message: {e}")
    return None


def append_dict(path, t, d, values):
  group = values[path]
  group["t"].append(t)

  # Determine all keys seen so far for this message type
  all_keys = set(group.keys()) | set(d.keys())
  all_keys.discard("t")

  # Ensure all fields have a value (or None) for this timestamp
  for k in all_keys:
    # If key is new, backfill previous timestamps with None
    if k not in group:
      group[k] = [None] * (len(group["t"]) - 1)
    # Append current value, or None if missing in this message
    group[k].append(d.get(k))


def potentially_ragged_array(arr, dtype=None, **kwargs):
  if arr and isinstance(arr[0], str):
    return np.array(arr, dtype=object, **kwargs)
  try:
    return np.array(arr, dtype=dtype, **kwargs)
  except ValueError:
    return np.array(arr, dtype=object, **kwargs)


def _should_expand_ragged_field(ragged_values: np.ndarray) -> bool:
  # Find the first non-None entry to inspect its type
  first_entry = next((entry for entry in ragged_values if entry is not None), None)

  if first_entry is None or isinstance(first_entry, str) or not hasattr(first_entry, '__len__'):
    return False

  # Don't expand numpy arrays with non-numeric types
  if hasattr(first_entry, 'dtype') and first_entry.dtype.kind in ['U', 'S', 'O']:
    return False

  return True


def _expand_ragged_field(base_path: str, timestamps: np.ndarray, ragged_values: np.ndarray) -> dict:
  result = {}
  # Find the maximum length across all list-like entries
  max_len = max((len(e) for e in ragged_values if hasattr(e, '__len__') and not isinstance(e, str)), default=0)

  for i in range(max_len):
    # Use a comprehension to gather valid (ts, val) pairs for the current index
    ts_val_pairs = [
      (timestamps[t_idx], entry[i]) for t_idx, entry in enumerate(ragged_values) if hasattr(entry, '__len__') and not isinstance(entry, str) and len(entry) > i
    ]

    if ts_val_pairs:
      # Unzip pairs into separate timestamp and value lists
      index_timestamps, index_values = zip(*ts_val_pairs)
      field_path = f"{base_path}/{i}"
      result[field_path] = {'timestamps': np.array(index_timestamps, dtype=np.float64), 'values': potentially_ragged_array(list(index_values))}

  return result


def msgs_to_time_series(msgs: list) -> dict[str, dict[str, np.ndarray]]:
  """
  Extract all scalar fields from capnp messages into individual time series.
  """
  intermediate_values = defaultdict(lambda: {"t": []})
  for msg in msgs:
    typ = msg.which()
    sub_msg = getattr(msg, typ)

    if (msg_dict := get_message_dict(sub_msg, typ)) is not None:
      tm = msg.logMonoTime / 1.0e9
      msg_dict['_valid'] = msg.valid
      append_dict(typ, tm, msg_dict, intermediate_values)

  # Convert collected lists to sorted numpy arrays
  for group in intermediate_values.values():
    if not group["t"]:
      continue
    order = np.argsort(group["t"])
    for name, group_values in group.items():
      group[name] = potentially_ragged_array(group_values)[order]

  # Second pass: convert to individual field time series
  result = {}
  for msg_type, data in intermediate_values.items():
    if "t" not in data or len(data["t"]) == 0:
      continue
    timestamps = data.pop("t")
    for field_name, field_values in data.items():
      field_path = f"{msg_type}/{field_name}"

      if field_values.dtype == object and _should_expand_ragged_field(field_values):
        result.update(_expand_ragged_field(field_path, timestamps, field_values))
      else:
        result[field_path] = {'timestamps': timestamps, 'values': field_values}

  return result


class DataSource(ABC):
  @abstractmethod
  def load_data(self) -> dict[str, Any]:
    pass

  @abstractmethod
  def get_duration(self) -> float:
    pass


class LogReaderSource(DataSource):
  def __init__(self, route_name: str):
    self.route_name = route_name
    self._duration = 0.0
    self._start_time_mono = 0.0

  def load_data(self) -> dict[str, Any]:
    lr = LogReader(self.route_name)
    processed_data = msgs_to_time_series(lr)

    min_time = float('inf')
    max_time = float('-inf')

    for data in processed_data.values():
      if len(data['timestamps']) > 0:
        min_time = min(min_time, data['timestamps'][0])
        max_time = max(max_time, data['timestamps'][-1])

    if min_time != float('inf'):
      self._start_time_mono = min_time
      self._duration = max_time - min_time
    else:
      self._start_time_mono = 0.0
      self._duration = 0.0

    return {'time_series_data': processed_data, 'route_start_time_mono': self._start_time_mono, 'duration': self._duration}

  def get_duration(self) -> float:
    return self._duration


class DataLoadedEvent:
  def __init__(self, data: dict[str, Any]):
    self.data = data


class Observer(ABC):
  @abstractmethod
  def on_data_loaded(self, event: DataLoadedEvent):
    pass


class DataManager:
  def __init__(self):
    self.time_series_data: dict[str, dict[str, np.ndarray]] = {}
    self.loading = False
    self.route_start_time_mono = 0.0
    self.duration = 0.0
    self._observers: list[Observer] = []

  def add_observer(self, observer: Observer):
    self._observers.append(observer)

  def remove_observer(self, observer: Observer):
    if observer in self._observers:
      self._observers.remove(observer)

  def _notify_observers(self, event: DataLoadedEvent):
    for observer in self._observers:
      observer.on_data_loaded(event)

  def get_current_value_for_path(self, path: str, time_s: float, last_index: int | None = None):
    try:
      abs_time_s = self.route_start_time_mono + time_s
      ts_data = self.time_series_data.get(path)

      if ts_data is None:
        return None, None

      t, v = ts_data['timestamps'], ts_data['values']

      if len(t) == 0:
        return None, None

      if last_index is None:  # jump
        idx = np.searchsorted(t, abs_time_s, side='right') - 1
      else:  # continuous playback
        idx = last_index
        while idx < len(t) - 1 and t[idx + 1] < abs_time_s:
          idx += 1

      idx = max(0, idx)
      return v[idx], idx

    except (KeyError, IndexError):
      return None, None

  def get_all_paths(self) -> list[str]:
    return list(self.time_series_data.keys())

  def is_path_plottable(self, path: str) -> bool:
    ts_data = self.time_series_data.get(path)
    if ts_data is not None:
      value_array = ts_data.get('values')
      if value_array is not None and value_array.size > 0:
        # only numbers and bools are plottable
        return np.issubdtype(value_array.dtype, np.number) or np.issubdtype(value_array.dtype, np.bool_)
    return False

  def get_time_series_data(self, path: str) -> tuple | None:
    ts_data = self.time_series_data.get(path)
    if ts_data is None:
      return None

    try:
      time_array = ts_data['timestamps']
      plot_values = ts_data['values']

      if len(time_array) == 0:
        return None

      rel_time_array = time_array - self.route_start_time_mono
      return rel_time_array, plot_values

    except KeyError:
      return None

  def load_route(self, route_name: str):
    if self.loading:
      return

    self.loading = True
    data_source = LogReaderSource(route_name)
    threading.Thread(target=self._load_in_background, args=(data_source,), daemon=True).start()

  def _load_in_background(self, data_source: DataSource):
    try:
      data = data_source.load_data()
      self.time_series_data = data['time_series_data']
      self.route_start_time_mono = data['route_start_time_mono']
      self.duration = data['duration']

      self._notify_observers(DataLoadedEvent(data))

    except Exception:
      cloudlog.exception("Error loading route:")
    finally:
      self.loading = False
