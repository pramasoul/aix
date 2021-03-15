from celery import Celery

import numpy as np
from nnbench import NetMaker, NNMEG
import time
import neo4j
import tools.neotools as nj
import grexutils as gu

driver = neo4j.GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "test"))
driver.verify_connectivity()

app = Celery('tasks',
             backend='rpc://',
             broker='pyamqp://guest@rabbitmq//')

@app.task
def add(x, y):
    q = """
MERGE (r:celeres {op: 'add', x: $x, y: $y, result: $result})
"""
    nj.query_write(driver, q, x=x, y=y, result=x+y)

@app.task
def div(dividend, divisor):
    q = """
MERGE (r:celeres
{op: 'div', dividend: $dividend, divisor: $divisor, result: $result})
"""
    nj.query_write(driver, q, dividend=dividend, divisor=divisor, result=dividend/divisor)


@app.task
def run_epp_code_strings(experiment_unikey, procedure_unikey, parameters_unikey):
    code_strings = gu.get_code_strings_of_experiment_procedure_parameters(driver,
                                                                                experiment_unikey=experiment_unikey,
                                                                                procedure_unikey=procedure_unikey,
                                                                                parameters_unikey=parameters_unikey)
    gu.now_run_it(driver, code_strings)