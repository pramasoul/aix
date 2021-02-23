### WIP ###

from typing import Any, Optional

from rx.core import typing
from rx.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable

#from ..periodicscheduler import PeriodicScheduler
from rx.scheduler.periodicscheduler import PeriodicScheduler

class TrioScheduler(PeriodicScheduler):
    """A scheduler that schedules work via the Trio main event loop.

    """

    def __init__(self) -> None:
        """Create a new TrioScheduler.

        """
        import trio

        super().__init__()


    def schedule(self,
                 action: typing.ScheduledAction,
                 state: Optional[typing.TState] = None
                 ) -> typing.Disposable:
        """Schedules an action to be executed.

        Args:
            action: Action to be executed.
            state: [Optional] state to be given to the action function.

        Returns:
            The disposable object used to cancel the scheduled action
            (best effort).
        """

        return self.schedule_relative(0.0, action, state)

    def schedule_relative(self,
                          duetime: typing.RelativeTime,
                          action: typing.ScheduledAction,
                          state: Optional[typing.TState] = None
                          ) -> typing.Disposable:
        """Schedules an action to be executed after duetime.

        Args:
            duetime: Relative time after which to execute the action.
            action: Action to be executed.
            state: [Optional] state to be given to the action function.

        Returns:
            The disposable object used to cancel the scheduled action
            (best effort).
        """

        sad = SingleAssignmentDisposable()

        def invoke_action() -> None:
            sad.disposable = self.invoke_action(action, state=state)

        msecs = max(0, int(self.to_seconds(duetime) * 1000.0))
        #timer = self._root.after(msecs, invoke_action)
        ### Hacking ***

        print("yo")
        async def run_after(msecs, invoke_action):
            await trio.sleep(msecs/1000)
            invoke_action()

        async def get_it_going():
            async with trio.open_nursery() as nursery:
                nursery.start_soon(run_after)

        # This assumes we are in Jupyter running under `%autoawait trio`
        #await get_it_going()

        def dispose() -> None:
            #self._root.after_cancel(timer)
            pass

        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_absolute(self,
                          duetime: typing.AbsoluteTime,
                          action: typing.ScheduledAction,
                          state: Optional[typing.TState] = None
                          ) -> typing.Disposable:
        """Schedules an action to be executed at duetime.

        Args:
            duetime: Absolute time at which to execute the action.
            action: Action to be executed.
            state: [Optional] state to be given to the action function.

        Returns:
            The disposable object used to cancel the scheduled action
            (best effort).
        """

        duetime = self.to_datetime(duetime)
        return self.schedule_relative(duetime - self.now, action, state=state)
