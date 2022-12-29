# reader states here are reduced only on snapshot save
# during iterations only gpu's progress stored
import json
import logging

from common.dictionary import LettersDict, SentencePieceDict
from common.disk_utils import ReaderProgress, ReadThreadState

from collections import defaultdict

LOG = logging.getLogger()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, LettersDict):
            return {"type": "LettersDict", "letters": obj._letters}
        elif isinstance(obj, SentencePieceDict):
            return {"type": "SentencePieceDict", "dict_file": obj._dict_file}
        return super().default(obj)


class ProgressTracker:
    def __init__(self):
        self._params_history = []
        self._data_sources = []
        self._reader_states = []
        self._epoch_sizes = []
        self._epoch_source_ids = []
        self._epoch = 0
        self._epoch_not_finished = False
        self._total_iterations = 0
        self._log_dir = None
        self._log_data = defaultdict(list)

        # epochs in each train
        self._training_sizes = []
        self._training_not_finished = False

    def is_training_finished(self):
        return not self._training_not_finished

    def last_training_params(self):
        return self._params_history[-1]

    def on_new_data_sources(self, data_sources):
        if self._training_not_finished or self._epoch_not_finished:
            raise Exception("Finish previous training first")

        self._data_sources.append(data_sources)
        self._reader_states.append(ReaderProgress())

    def on_new_training(self, params):
        LOG.info(f"Start new training with params: {json.dumps(params, cls=CustomJSONEncoder)}")
        self._params_history.append(params)
        if self._training_not_finished or self._epoch_not_finished:
            raise Exception("Finish previous training first")
        self._training_not_finished = True

    def start_epoch(self):
        if not self._epoch_not_finished:
            self._epoch += 1
            self._epoch_sizes.append(0)
            self._epoch_source_ids.append(len(self._data_sources) - 1)
            self._epoch_not_finished = True
        else:
            raise Exception("Can't start epoch second time")

    # TODO: automatic serialization for fields, this is ugly
    def save_distributed(self):
        from common.train_utils import reduce_reader_progresses
        reduced_reader_states = reduce_reader_progresses(self._reader_states, self.iteration())
        state = {
            "params_history": self._params_history,
            "data_sources": self._data_sources,
            "reader_states": [reader_state.save() for reader_state in reduced_reader_states],
            "epoch_sizes": self._epoch_sizes,
            "epoch_source_ids": self._epoch_source_ids,
            "epoch": self._epoch,
            "epoch_not_finished": self._epoch_not_finished,
            "total_iterations": self._total_iterations,
            "training_sizes": self._training_sizes,
            "training_not_finished": self._training_not_finished,
            "log_data": self._log_data
        }
        return state

    def add_scalar(self, name, value):
        iter_num = self.total_iterations()
        self._log_data[name].append((iter_num, value))

    def load(self, state):
        self._params_history = state["params_history"]
        self._data_sources = state["data_sources"]
        self._reader_states = [ReaderProgress().set_progress(reader_state) for reader_state in state["reader_states"]]
        self._epoch_sizes = state["epoch_sizes"]
        self._epoch_source_ids = state["epoch_source_ids"]
        self._epoch = state["epoch"]
        self._epoch_not_finished = state["epoch_not_finished"]
        self._total_iterations = state["total_iterations"]
        self._log_dir = state.get("log_dir")
        self._training_sizes = state["training_sizes"]
        self._training_not_finished = state["training_not_finished"]
        self._log_data = state["log_data"]

    def is_current_epoch_finished(self):
        return not self._epoch_not_finished

    def finish_batch(self):
        assert self._epoch_not_finished
        self._epoch_sizes[-1] += 1
        self._total_iterations += 1

    def finish_read_block(self, reader_state: ReadThreadState, block_seconds):
        assert self._epoch_not_finished
        self._reader_states[-1].update(reader_state)

    def epoch(self):
        return self._epoch

    def epoch_reader_state(self):
        return self._reader_states[-1]

    def is_same_data_sources(self, data_source):
        return self._data_sources[-1] == data_source if len(self._data_sources) > 0 else False

    def total_iterations(self):
        return self._total_iterations

    def log_dir(self):
        return self._log_dir

    # current step in epoch
    def iteration(self):
        return self._epoch_sizes[-1] if len(self._epoch_sizes) else 0

    def finish_epoch(self):
        self._epoch_not_finished = False

    def finish_training(self):
        if self._epoch_not_finished:
            raise Exception("finish epoch first")
        self._training_not_finished = False
        self._training_sizes.append({"total_iterations": self._total_iterations, "total_epochs": self._epoch})

    def previous_training_epochs(self):
        if len(self._training_sizes) > 0:
            return self._training_sizes[-1]["total_epochs"]
        else:
            return 0

    def previous_training_iterations(self):
        if len(self._training_sizes) > 0:
            return self._training_sizes[-1]["total_iterations"]
        else:
            return 0
