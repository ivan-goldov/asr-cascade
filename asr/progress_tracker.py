from common.progress_tracker import ProgressTracker
from common.disk_utils import ReadThreadState

import sys


class ASRProgressTracker(ProgressTracker):
    def __init__(self):
        super().__init__()
        self._epoch_sizes_seconds = []

    def start_epoch(self):
        super().start_epoch()
        self._epoch_sizes_seconds.append(0)

    def save_distributed(self):
        state = super().save_distributed()
        state["epoch_size_seconds"] = self._epoch_sizes_seconds
        return state

    def load(self, state):
        super().load(state)
        self._epoch_sizes_seconds = state["epoch_size_seconds"]

    def finish_read_block(self, reader_state: ReadThreadState, block_seconds):
        sys.stderr.write('ASRProgressTracker: finish_read_block\n')
        assert self._epoch_not_finished
        sys.stderr.write('ASRProgressTracker: finish_read_block before update reader_state\n')
        # self._reader_states[-1].update(reader_state) # Commented because of rudity
        sys.stderr.write('ASRProgressTracker: finish_read_block before epoch sizes seconds += block_seconds\n')
        self._epoch_sizes_seconds[-1] += block_seconds

    def epoch_duration(self):
        return self._epoch_sizes_seconds[-1] if len(self._epoch_sizes_seconds) else 0
