from wit_world.exports.cacheback import CacheTable as CacheTableBase, DraftResult
from lru_cache import TwoLevelLRUCache


class TrieNode:
    __slots__ = ("children", "token", "position")

    def __init__(self, position: int, token: int) -> None:
        self.children: list["TrieNode"] = []
        self.token = token
        self.position = position


class TrieForest:
    __slots__ = ("roots", "root_position")

    def __init__(self, root_position: int) -> None:
        self.roots: list[TrieNode] = []
        self.root_position = root_position

    def insert(self, tokens: list[int], positions: list[int]) -> None:
        if not tokens or not positions or positions[0] != self.root_position:
            return
        nodes = self.roots
        for tok, pos in zip(tokens, positions):
            found = None
            for n in nodes:
                if n.position == pos and n.token == tok:
                    found = n
                    break
            if found is None:
                found = TrieNode(pos, tok)
                nodes.append(found)
            nodes = found.children

    def linearize(self) -> tuple[list[int], list[int]]:
        tokens: list[int] = []
        positions: list[int] = []

        def dfs(node: TrieNode) -> None:
            tokens.append(node.token)
            positions.append(node.position)
            for child in node.children:
                dfs(child)

        for root in self.roots:
            dfs(root)
        return tokens, positions


class CacheTable(CacheTableBase):
    def __init__(self, leader_capacity, follower_capacity, leader_len, follower_len):
        self._cache = TwoLevelLRUCache(
            leader_capacity, follower_capacity, leader_len, follower_len
        )
        window_len = leader_len + follower_len - 1
        self._prev_window = [0] * window_len

    def update(self, context):
        leader_len = self._cache._leader_len
        follower_len = self._cache._follower_len
        total = leader_len + follower_len
        window_len = total - 1

        full = self._prev_window + list(context)

        if len(full) >= total:
            for i in range(len(full) - total + 1):
                leader = tuple(full[i : i + leader_len])
                follower = tuple(full[i + leader_len : i + total])
                self._cache.put(leader, follower)

        self._prev_window = full[-window_len:]

    def draft(self):
        leader_len = self._cache._leader_len
        follower_len = self._cache._follower_len
        positions = list(range(1, follower_len + 1))
        trie = TrieForest(1)

        key = tuple(self._prev_window[-leader_len:])
        followers = self._cache.get(key)
        if followers is not None:
            for f in followers:
                trie.insert(list(f), positions)

        tokens, pos = trie.linearize()
        return DraftResult(tokens=tokens, positions=pos)

    def clear(self):
        self._cache.clear()
        self._prev_window = [0] * len(self._prev_window)
