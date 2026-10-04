"""Overlap unchanged video readers with inference while preserving order."""
import os
from queue import Empty, Full, Queue
from threading import Event, Thread


def prefetch(iterator, capacity=2):
    queue = Queue(maxsize=capacity)
    stop = Event()
    finished = object()

    def put(item):
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.2)
                return True
            except Full:
                pass
        return False

    def produce():
        try:
            for value in iterator:
                if not put((True, value)):
                    break
        except BaseException as error:
            put((False, error))
        finally:
            close = getattr(iterator, 'close', None)
            if close:
                close()
            put((True, finished))

    thread = Thread(target=produce, name='stage2-frame-prefetch', daemon=True)
    thread.start()
    try:
        while True:
            try:
                success, value = queue.get(timeout=0.5)
            except Empty:
                if not thread.is_alive():
                    raise RuntimeError('Frame prefetch exited without completion')
                continue
            if not success:
                raise value
            if value is finished:
                return
            yield value
    finally:
        stop.set()
        thread.join(timeout=2)


def install(module):
    batch_capacity = max(1, int(os.environ.get('HIGHLIGHT_BATCH_PREFETCH', '2')))
    frame_capacity = max(1, int(os.environ.get('HIGHLIGHT_FRAME_PREFETCH', '16')))
    if hasattr(module, 'read_frame_batches'):
        original = module.read_frame_batches
        module.read_frame_batches = lambda *a, **kw: prefetch(original(*a, **kw), capacity=batch_capacity)
    if hasattr(module, 'read_sampled_frames'):
        original_sampled = module.read_sampled_frames
        module.read_sampled_frames = lambda *a, **kw: prefetch(original_sampled(*a, **kw), capacity=frame_capacity)
