#!/usr/bin/env python
# coding: utf-8

# # ReactiveX under trio

# In[1]:


import trio


# In[2]:


#get_ipython().run_line_magic('autoawait', 'trio')


# In[3]:


import rx


# In[4]:


from trioscheduler import TrioScheduler
from rx.scheduler.eventloop import AsyncIOThreadSafeScheduler


# In[6]:


def lowercase():
    def _lowercase(source):
        def subscribe(observer, scheduler = None):
            def on_next(value):
                observer.on_next(value.lower())

            return source.subscribe(
                on_next,
                observer.on_error,
                observer.on_completed,
                scheduler)
        return rx.create(subscribe)
    return _lowercase

#sch = rx.scheduler.TrampolineScheduler()
#sch = TrioScheduler()
#sch = AsyncIOThreadSafeScheduler
sch = 123

rx.of("Alpha", "Beta", "Gamma", "Delta", "Epsilon").pipe(
        observe_on(sch),
        lowercase()
     ).subscribe(lambda value: print("Received {0}".format(value)), scheduler=sch)


# In[ ]:




