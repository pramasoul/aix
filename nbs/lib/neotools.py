#!/usr/bin/env python
# coding: utf-8

# # `neo4j` Techniques

import neo4j
import numpy as np
import functools
from hashlib import sha256

driver = neo4j.GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "nonstandard"))

# ## A generic transaction function that calls back with each record
def tx_fun(tx, query, callback, **kwargs):
    for record in tx.run(query, **kwargs):
        callback(record)


# ## Transaction doers
def do_a_tx(driver, query, callback, r_or_w, **kwargs):
    with driver.session() as session:
        if r_or_w == "r":
            return session.read_transaction(tx_fun, query, callback, **kwargs)
        else:
            return session.write_transaction(tx_fun, query, callback, **kwargs)


do_a_read_tx = functools.partial(do_a_tx, r_or_w="r")
do_a_write_tx = functools.partial(do_a_tx, r_or_w="w")

# ## Query functions
def query_return_list(driver, query, r_or_w, **kwargs):
    records = []

    def gather_records(r):
        records.append(r)

    do_a_tx(driver, query, gather_records, r_or_w, **kwargs)
    return records


query_read_return_list = functools.partial(query_return_list, r_or_w="r")
query_write_return_list = functools.partial(query_return_list, r_or_w="w")

# Write, expecting (and asserting) no reponse records
def query_write(driver, query, **kwargs):
    rl = query_write_return_list(driver, query, **kwargs)
    assert (
        rl == []
    ), f"Did you expect some response to your query?\n\tGot {rl} from query {query}"


# ### Yield records from query
# We'd like multi-record queries to yield each record as it comes from the database.
# I don't know how to do that yet. This function invites applications to structure for
# it working as it should eventually.
def query_yield(driver, query, r_or_w, **kwargs):
    for r in query_return_list(driver, query, r_or_w, **kwargs):
        yield r


query_read_yield = functools.partial(query_yield, r_or_w="r")
query_write_yield = functools.partial(query_yield, r_or_w="w")

# Store and retrieve numpy arrays in the graph database
class NumpyStore:
    def __init__(self, driver):
        self.driver = driver

    @staticmethod
    def _np_to_key(a):
        m = sha256(a)
        m.update(a.dtype.name.encode("utf8"))
        m.update(str(a.shape).encode("utf8"))
        return m.hexdigest()[:16]

    @staticmethod
    def _add_np(tx, a, k):
        tx.run(
            "MERGE (:ndarray {k: $k, dtype: $dtype, shape: $shape, bytes: $bytes})",
            k=k,
            dtype=a.dtype.name,
            shape=list(a.shape),
            bytes=a.tobytes(),
        )

    @staticmethod
    def _get_np(tx, k):
        response = tx.run(
            "MATCH (a:ndarray) WHERE a.k = $k "
            "RETURN a.dtype as dtype, a.shape as shape, a.bytes as bytes",
            k=k,
        )
        n_matches = 0
        for r in response:
            n_matches += 1
            a = np.frombuffer(r["bytes"], dtype=r["dtype"]).reshape(r["shape"])
        assert (
            n_matches <= 1
        ), f"Found {n_matches} arrays of key {k}, when should only be one."
        if n_matches < 1:
            raise KeyError
        else:
            return a

    @staticmethod
    def _dedup(tx):
        q = """
MATCH (n:ndarray)
WITH n.k as k, collect(n) AS nds
WHERE size(nds) > 1
FOREACH (n in tail(nds) | DELETE n)
"""
        tx.run(q)

    def _tput(self, a):
        k = self._np_to_key(a)
        with self.driver.session() as session:
            session.write_transaction(self._add_np, a, k=k)
        return k

    def _tget(self, k):
        with self.driver.session() as session:
            a = session.read_transaction(self._get_np, k)
        assert k == self._np_to_key(a)
        return a

    def store(self, a):
        return self._tput(a)

    def retrieve(self, key):
        return self._tget(k)

    def dedup(self):
        with self.driver.session() as session:
            session.write_transaction(self._dedup)

    def __getitem__(self, k):
        return self._tget(k)

    def __setitem__(self, k, v):
        return self._tput(v)

    def __delitem__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
