import itertools
from heapq import heappush, heappop


class PriorityQueue:
    """'priority' follows the convention that larger value = higher priority"""

    def __init__(self):
        self.pq = []  # list of entries arranged in a min heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = "<removed-task>"  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def push(self, task, priority=0):
        "Add a new task or update the priority of an existing task"
        if task in self.entry_finder:
            self.remove(task)
        count = next(self.counter)

        # negate priority so we can use the minheap in heapq
        entry = [-priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove(self, task):
        "Mark an existing task as REMOVED.  Raise KeyError if not found."
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop(self, return_priority=False):
        "Remove and return the highest priority task. Raise KeyError if empty."
        while self.pq:
            neg_priority, _, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                if return_priority:
                    return task, -neg_priority
                else:
                    return task
        raise KeyError("pop from an empty priority queue")

    def top(self):
        while self.pq:
            neg_priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                heappush(self.pq, [neg_priority, count, task])
                return task
        raise KeyError("top_task from an empty priority queue")

    def __len__(self):
        # more sophisticated - doesn't count "removed" items
        # return len([t for p, c, t in self.pq if t is not self.REMOVED])
        return len(self.pq)

    def __getitem__(self, index):
        return self.pq[index]
