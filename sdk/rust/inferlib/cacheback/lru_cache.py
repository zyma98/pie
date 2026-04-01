from collections import OrderedDict
from typing import Optional

type Tokens = tuple[int, ...]


class TwoLevelLRUCache:
    """Two-level LRU cache.

    - Top level: up to `leader_capacity` distinct leaders (each a tuple[int, ...]).
      Most-recently-used (MRU) leader is on the right; least-recently-used (LRU)
      leader on the left.
    - Second level: for every leader, up to `follower_capacity` followers
      (also tuples[int, ...]), kept in their own per-leader LRU list.

    All public ops are O(1):
    - put(leader, follower):       insert / update and mark MRU
    - get(leader):                 return all followers for leader and mark leader MRU
    - get_follower(leader, follower): check specific follower, mark both levels MRU
    - __contains__(leader):        membership test
    - __len__():                   number of leaders currently held
    """

    def __init__(
        self,
        leader_capacity: int,
        follower_capacity: int,
        leader_len: int,
        follower_len: int,
    ) -> None:
        if leader_capacity <= 0 or follower_capacity <= 0:
            raise ValueError("Capacities must be positive integers")
        self._leader_capacity = leader_capacity
        self._follower_capacity = follower_capacity
        self._leader_len = leader_len
        self._follower_len = follower_len

        # OrderedDict[leader, OrderedDict[follower, None]]
        self._cache: OrderedDict[Tokens, OrderedDict[Tokens, None]] = OrderedDict()

    def _touch_leader(self, leader: Tokens) -> None:
        """Mark `leader` as most-recently used."""
        self._cache.move_to_end(leader, last=True)

    def _touch_follower(self, leader: Tokens, follower: Tokens) -> None:
        """Mark `follower` (under `leader`) as most-recently used."""
        self._cache[leader].move_to_end(follower, last=True)

    def put(self, leader: Tokens, follower: Tokens) -> None:
        """Insert `follower` for `leader` (or refresh its recency if already present).

        Handles both leader- and follower-level eviction when capacity limits
        are exceeded.
        """
        if leader in self._cache:
            self._touch_leader(leader)
            vcache = self._cache[leader]
            if follower in vcache:
                self._touch_follower(leader, follower)
            else:
                if len(vcache) >= self._follower_capacity:
                    vcache.popitem(last=False)
                vcache[follower] = None
                self._touch_follower(leader, follower)
        else:
            if len(self._cache) >= self._leader_capacity:
                self._cache.popitem(last=False)
            self._cache[leader] = OrderedDict({follower: None})

    def get(self, leader: Tokens) -> Optional[list[Tokens]]:
        """Return all followers associated with `leader` (MRU to LRU order) and
        mark `leader` as most-recently used.  Returns None on a miss.
        """
        if leader not in self._cache:
            return None
        self._touch_leader(leader)
        return list(self._cache[leader].keys())

    def get_follower(self, leader: Tokens, follower: Tokens) -> Optional[Tokens]:
        """Access a specific (leader, follower) pair.

        Touches both leader and follower on a hit; returns the `follower` or
        None on a miss.
        """
        if leader not in self._cache:
            return None
        vcache = self._cache[leader]
        if follower not in vcache:
            self._touch_leader(leader)
            return None
        self._touch_leader(leader)
        self._touch_follower(leader, follower)
        return follower

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def __contains__(self, leader: Tokens) -> bool:
        return leader in self._cache

    def __len__(self) -> int:
        """Number of leaders currently stored."""
        return len(self._cache)
