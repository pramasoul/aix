#!/usr/bin/env python
# coding: utf-8

# # Graph Experiment utilities

from collections import defaultdict
from more_itertools import collapse, flatten, groupby_transform
import numpy as np
from nnbench import netMaker
from operator import itemgetter
import time

import neo4j
import tools.neotools as nj


# https://neo4j.com/docs/cypher-manual/current/styleguide/

class Thing:
    """A generic object to hold attributes"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Methods to create nodes in the experiment graph

def create_an_experiment(driver, **kwargs):
    q = """
CREATE (e:Experiment {name: $experiment_name,
                    unikey: $experiment_unikey,
                        ts: timestamp()})
"""
    nj.query_write(driver, q, **kwargs)

def create_a_procedure(driver, **kwargs):
    q = """
MATCH (e:Experiment {unikey: $experiment_unikey})
CREATE (e)-[:INCLUDES]->
(:Procedure {name: $procedure_name,
           unikey: $procedure_unikey,
     code_strings: $code_strings,
               ts: timestamp()})
"""
    nj.query_write(driver, q, **kwargs)

def create_parameters_to_experiment_procedure(driver, **kwargs):
    q = """
MATCH (proc:Procedure {unikey: $procedure_unikey})
CREATE (proc)-[:INCORPORATES]->
(par:Parameters {unikey: $parameters_unikey,
                   name: $parameters_name,
   prepend_code_strings: $prepend_code_strings,
    append_code_strings: $append_code_strings,
                  trial: $trial,
                     ts: timestamp()})
"""
    nj.query_write(driver, q, **kwargs)

# Functions to get Experiments:

def get_experiment_names_keys(driver):
    q="""
MATCH (e:Experiment)
RETURN e.name as name, e.unikey as key
"""
    return [(r['name'], r['key']) for r in nj.query_read_yield(driver, q)]

def get_experiment_key_from_name(driver, name):
    q="""
MATCH (e:Experiment {name: $name})
RETURN e.unikey as unikey
"""
    records = nj.query_read_return_list(driver, q, name=name)
    if len(records) < 1:
        raise KeyError(f'No experiment found with name "{name}"')
    if len(records) > 1:
        raise KeyError(f'Found {len(records)} experiments with name "{name}"')
    return records[0]['unikey']

# Functions to get Procedures:

def get_procedure_names_keys_from_experiment_key(driver, key):
    q = """
MATCH (:Experiment {unikey: $key})
-[:INCLUDES]->
(proc:Procedure)
RETURN proc.name as name, proc.unikey as key
"""
    return [(r['name'], r['key']) for r in nj.query_read_yield(driver, q, key=key)]

def parameter_names_keys_from_experiment_and_procedure_names(driver, ex_name, proc_name):
    q = """
MATCH (e:Experiment {name: $ex_name})
-[:INCLUDES]->(proc:Procedure {name: $proc_name})
-[:INCORPORATES]->(par:Parameters)
RETURN par.name as name, par.unikey as key
"""
    return [(r['name'], r['key']) for r in
            nj.query_read_yield(driver, q, ex_name=ex_name, proc_name=proc_name)]

def parameter_names_keys_from_experiment_and_procedure_keys(driver, ex_key, proc_key):
    q = """
MATCH (e:Experiment {unikey: $ex_key})
-[:INCLUDES]->(proc:Procedure {unikey: $proc_key})
-[:INCORPORATES]->(par:Parameters)
RETURN par.name as name, par.unikey as key
"""
    return [(r['name'], r['key']) for r in
            nj.query_read_yield(driver, q, ex_key=ex_key, proc_key=proc_key)]

def get_unstarted_parameters_of_procedure(driver, **kwargs):
    q="""
MATCH (:Procedure {unikey: $procedure_unikey})
-[:INCORPORATES]->
(par:Parameters)
WHERE NOT (par)-[:CONFIGURES]->(:Net)
RETURN par.unikey as unikey
"""
    return [r['unikey'] for r in nj.query_read_yield(driver, q, **kwargs)]

def get_code_strings_of_experiment_procedure_parameters(driver, **kwargs):
    q = """
MATCH (e:Experiment {unikey: $experiment_unikey})
-[:INCLUDES]->
(proc:Procedure {unikey: $procedure_unikey})
-[:INCORPORATES]->
(par:Parameters {unikey: $parameters_unikey})
RETURN par.prepend_code_strings, proc.code_strings, par.append_code_strings
"""
    records = nj.query_read_return_list(driver, q, **kwargs)
    if len(records) < 1:
        raise KeyError(f'No experiment,procedure,parameters found to match "{kwargs}"')
    if len(records) > 1:
        raise KeyError(f'Found {len(records)} experiment,procedure,parameters "{kwargs}"')
    r = records[0]
    #print(f"r['proc.code_strings'] = {r['proc.code_strings']}")
    #print(f"r['par.code_strings'] = {r['par.code_strings']}")
    return r['par.prepend_code_strings'] + \
            r['proc.code_strings'] + \
            r['par.append_code_strings']

def get_code_strings_from_db(driver, experiment_name, procedure_name):
    experiment_unikey = get_experiment_key_from_name(driver, name=experiment_name)
    #print(experiment_unikey)
    procedures = dict(get_procedure_names_keys_from_experiment_key(driver, key=experiment_unikey))
    procedure_unikey = procedures[procedure_name]
    #print(procedure_unikey)
    unstarted_parameters = get_unstarted_parameters_of_procedure(driver, procedure_unikey=procedure_unikey)
    parameters_unikey = unstarted_parameters[0]
    #print(parameters_unikey)
    code_strings_from_db = get_code_strings_of_experiment_procedure_parameters(driver,
        experiment_unikey=experiment_unikey,
        procedure_unikey=procedure_unikey,
        parameters_unikey=parameters_unikey)
    return code_strings_from_db

def now_run_it(driver, code_strings):
    cx = {}
    for s in code_strings:
        exec(s, cx)
    nps = nj.NumpyStore(driver)
    cx['run_it'](cx, driver, nps)

################################################################################
# python Object representations of graph nodes

class UnikeyedObjectProperty:
    def __set_name__(self, owner, attr_name):
        #print(owner.__name__)
        self.owner = owner
        self.attr_name = attr_name
        self.fetch_q ="MATCH (n:%s {unikey: $unikey}) RETURN n.%s as %s" % (owner.__name__, attr_name, attr_name)
        self.store_q = ""

    def __get__(self, obj, objtype=None):
        return nj.query_read_return_list(obj.driver, self.fetch_q, unikey=obj.unikey)[0][self.attr_name]

    def __set__(self, obj, value):
        raise NotImplementedError

class Experiment:
    name = UnikeyedObjectProperty()
    ts = UnikeyedObjectProperty()

    def __init__(self, driver, unikey):
        self.driver = driver
        self.unikey = unikey

class Procedure:
    name = UnikeyedObjectProperty()
    ts = UnikeyedObjectProperty()
    code_strings = UnikeyedObjectProperty()

    def __init__(self, driver, unikey):
        self.driver = driver
        self.unikey = unikey

class NNetProperty:
    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name
        self.cache_name = '_' + name
        #self.fetch_q = "MATCH (n:Net {unikey: $unikey}) RETURN n.ksv as ksv, n.loss as loss"

    def __get__(self, obj, objtype=None):
        if not hasattr(obj, self.cache_name):
            #print(f"creating {self.name}")
            #records = nj.query_read_return_list(obj.driver, self.fetch_q, unikey=obj.unikey)
            #if len(records) < 1:
            #    raise KeyError(f'No Net found with unikey={obj.unikey}')
            #if len(records) > 1:
            #    raise KeyError(f'Found {len(records)} Nets with unikey={obj.unikey}')
            #r = records[0]
            val = netMaker(obj.shorthand)
            val.set_state_from_vector(obj.nps[obj.ksv])
            setattr(obj, self.cache_name, val)
        val = getattr(obj, self.cache_name)
        return val

    def __set__(self, obj, value):
        raise NotImplementedError

class Net:
    batches_from_start = UnikeyedObjectProperty()
    head = UnikeyedObjectProperty()
    ksv = UnikeyedObjectProperty()
    loss = UnikeyedObjectProperty()
    shorthand = UnikeyedObjectProperty()
    ts = UnikeyedObjectProperty()
    nnet = NNetProperty()

    def __init__(self, driver, nps, unikey):
        self.driver = driver
        self.nps = nps
        self.unikey = unikey

class DescendentNetsProperty:
    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name
        self.cache_name = '_' + name
        self.fetch_q ="""
MATCH (:Parameters {unikey: $unikey})-[:CONFIGURES]->(head:Net)
MATCH p=(head)-[:LEARNED*]->(tail:Net)
WHERE NOT (tail)-->()
RETURN [n IN nodes(p) | n.unikey] as net_unikeys"""

    def __get__(self, obj, objtype=None):
        if not hasattr(obj, self.cache_name):
            #print(f"creating {self.name}")
            records = nj.query_read_return_list(obj.driver, self.fetch_q, unikey=obj.unikey)
            if len(records) < 1:
                #raise KeyError(f'No Nets found from Parameter with unikey={obj.unikey}')
                return []
            if len(records) > 1:
                raise KeyError(f'Found {len(records)} from Parameter with unikey={obj.unikey}')
            r = records[0]
            val = [Net(obj.driver, obj.nps, key) for key in r['net_unikeys']]
            setattr(obj, self.cache_name, val)
        val = getattr(obj, self.cache_name)
        return val

    def __set__(self, obj, value):
        raise NotImplementedError

class TrajectoryProperty:
    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name
        self.cache_name = '_' + name
        self.fetch_q ="""
MATCH (:Parameters {unikey: $unikey})-[:CONFIGURES]->(head:Net)
MATCH p=(head)-[:LEARNED*]->(tail:Net)
WHERE NOT (tail)-->()
RETURN relationships(p) AS relationships"""
        self.fetch_q ="""
MATCH (:Parameters {unikey: $unikey})-[:CONFIGURES]->(head:Net)
MATCH (head)-[t:LEARNED*]->(tail:Net)
WHERE NOT (tail)-->()
RETURN [r IN t | r.batch_points] AS batch_points"""
        self.fetch_q ="""
MATCH (:Parameters {unikey: $unikey})-[:CONFIGURES]->(head:Net)
MATCH (head)-[t:LEARNED*]->(tail:Net)
WHERE NOT (tail)-->()
UNWIND [r IN t | r.batch_points] AS u1
UNWIND u1 as u2
RETURN collect(u2) as batch_points"""
        self.fetch_q ="""
MATCH (:Parameters {unikey: $unikey})-[:CONFIGURES]->(head:Net)
MATCH (head)-[t:LEARNED*]->(tail:Net)
WHERE NOT (tail)-->()
UNWIND [r IN t | [r.batch_points] AS w1, [r IN t | [r.traj_cos_sq_signeds] AS w2
UNWIND w1 as u1, w2 as u2
RETURN collect(u1) AS batch_points, collect(u2) AS traj_cos_sq_signeds"""
        self.fetch_q ="""
MATCH (:Parameters {unikey: $unikey})-[:CONFIGURES]->(head:Net)
MATCH (head)-[t:LEARNED*]->(tail:Net)
WHERE NOT (tail)-->()
RETURN t"""

    def __get__(self, obj, objtype=None):
        if not hasattr(obj, self.cache_name):
            #print(f"creating {self.name}")
            records = nj.query_read_return_list(obj.driver, self.fetch_q, unikey=obj.unikey)
            if len(records) < 1:
                #raise KeyError(f'No Nets found from Parameter with unikey={obj.unikey}')
                return []
            if len(records) > 1:
                raise KeyError(f'Found {len(records)} from Parameter with unikey={obj.unikey}')
            r = records[0][0]
            val = Thing(**dict(((k, list(v))
                         for k,v in
                         groupby_transform(
                             sorted(
                                 flatten(
                                     list(a.items())
                                     for a in r
                                 ),
                                 key=itemgetter(0)
                             ),
                             itemgetter(0),
                             itemgetter(1),
                             collapse,
                         ))))
            #val.r = r #DEBUG
            setattr(obj, self.cache_name, val)
        val = getattr(obj, self.cache_name)
        return val

    def __set__(self, obj, value):
        raise NotImplementedError

class Parameters:
    name = UnikeyedObjectProperty()
    ts = UnikeyedObjectProperty()
    prepend_code_strings = UnikeyedObjectProperty()
    append_code_strings = UnikeyedObjectProperty()
    trial = UnikeyedObjectProperty()
    results = DescendentNetsProperty()
    trajectory = TrajectoryProperty()

    def __init__(self, driver, nps, unikey):
        self.driver = driver
        self.nps = nps
        self.unikey = unikey

