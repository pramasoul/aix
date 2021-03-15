#!/usr/bin/env python
# coding: utf-8

# # Graph Experiment 2 utilities

import numpy as np
from nnbench import NetMaker, NNMEG
import time
import tools.neotools as nj

import neo4j
from collections import defaultdict


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

# ## Methods to acquire work

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

def get_unstarted_parameters_of_procedure(driver, **kwargs):
    q="""
MATCH (:Procedure {unikey: $procedure_unikey})
-[:INCORPORATES]->
(par:Parameters)
WHERE NOT (par)-[:CONFIGURES]->(:net)
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
    assert len(records) == 1
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
