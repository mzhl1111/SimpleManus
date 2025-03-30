# tool/planning.py
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Literal
from typing_extensions import Self


class StepStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Step:
    content: str
    status: StepStatus = StepStatus.NOT_STARTED

    def transition(self, new_status: StepStatus) -> None:
        """
        Transition the step to a new status.

        Args:
            new_status (StepStatus): The new status to set for the step

        Raises:
            ValueError: If an invalid status transition is attempted
        """
        valid_transitions = {
            StepStatus.NOT_STARTED: [StepStatus.IN_PROGRESS],
            StepStatus.IN_PROGRESS: [StepStatus.COMPLETED, StepStatus.NOT_STARTED],
            StepStatus.COMPLETED: [StepStatus.IN_PROGRESS]
        }

        if new_status not in valid_transitions.get(self.status, []):
            raise ValueError(f"Invalid status transition from {self.status} to {new_status}")

        self.status = new_status

    def __repr__(self) -> str:
        return f"<Step: {self.content} | {self.status.value}>"


@dataclass
class Plan:
    steps: List[Step] = field(default_factory=list)

    @classmethod
    def from_descriptions(cls, step_descriptions: List[str]) -> Self:
        steps = [Step(content=desc) for desc in step_descriptions]
        return cls(steps=steps)

    def get_steps(self) -> List[Step]:
        return self.steps

    def update_step_status(self, index: int, status: StepStatus) -> None:
        self.steps[index].transition(status)

    def __repr__(self) -> str:
        steps_repr = '\n'.join(str(s) for s in self.steps)
        return f'<Plan: \n{steps_repr}>'


_PLANNING_TOOL_DESCRIPTION = """
A planning tool that allows the agent to create and manage plans for solving complex tasks.
The tool provides functionality for creating plans, updating plan steps, and tracking progress.
"""


class PlanningTool(BaseModel):
    """
    A planning tool that allows the agent to create and manage plans for solving complex tasks.
    The tool provides functionality for creating plans, updating plan steps, and tracking progress.
    """

    name: str = "planning"
    description: str = _PLANNING_TOOL_DESCRIPTION
    plan: Plan = None

    def create_plan(self, steps: str) -> str:
        steps = steps.split(',')
        if (
                not steps
                or not isinstance(steps, list)
                or not all(isinstance(step, str) for step in steps)
        ):
            return "Create plan error: Parameter `steps` must be a non-empty list of strings for command: create"
        self.plan = Plan.from_descriptions(steps)
        return f"Plan created: \n {str(self.plan)}"

    def update_plan(self, steps: str) -> str:
        """
        Update the existing plan with a new list of steps.

        Args:
            steps (List[str]): A new list of plan steps.

        Returns:
            str: A message describing the plan update result.
        """
        # Validate input
        steps = steps.split(',')
        if not steps or not isinstance(steps, list) or not all(isinstance(step, str) for step in steps):
            return "Update plan error: Parameter `steps` must be a non-empty list of strings"

        # If no existing plan, create a new one
        if not hasattr(self, 'plan') or not self.plan:
            self.plan = Plan.from_descriptions(steps)
            return f"Plan created: \n {str(self.plan)}"

        # Create a new plan with updated steps
        new_plan = Plan.from_descriptions(steps)

        # Carry over statuses of matching steps
        for new_idx, new_step in enumerate(new_plan.get_steps()):
            # Try to find a matching step in the old plan
            matching_step = None
            for old_idx, old_step in enumerate(self.plan.get_steps()):
                if new_step.content == old_step.content:
                    matching_step = old_step
                    break

            # If a matching step is found, copy its status
            if matching_step:
                new_plan.steps[new_idx].status = matching_step.status

        # Reset status for any completed steps that are now located after new steps
        for old_idx, old_step in enumerate(self.plan.get_steps()):
            if old_step.status == StepStatus.COMPLETED:
                # Find the index of this old step in the new plan
                try:
                    new_idx = next(
                        i for i, new_step in enumerate(new_plan.get_steps()) if new_step.content == old_step.content)

                    # Check if this completed step appears after any new steps
                    if any(new_step.content not in [old.content for old in self.plan.get_steps()]
                           for new_step in new_plan.steps[:new_idx]):
                        # Reset the step to not_started
                        new_plan.steps[new_idx].status = StepStatus.NOT_STARTED
                except StopIteration:
                    # Step not found in new plan, so we ignore it
                    pass

        # Update the plan
        self.plan = new_plan

        return f"Plan updated: \n {str(self.plan)}"

    def update_status(self, statuses: str) -> str:
        """
        Update the status of all steps in the current plan.

        Args:
            statuses (List[str]): A list of statuses corresponding to plan steps.

        Returns:
            str: A message describing the status update result.
        """
        statuses = statuses.split(',')
        if not hasattr(self, 'plan') or not self.plan:
            return "Update status error: No active plan exists"

        # Check if number of statuses matches number of steps
        if len(statuses) != len(self.plan.steps):
            return f"Update status error: Number of statuses ({len(statuses)}) must match number of steps ({len(self.plan.steps)})"

        # Validate status values
        valid_statuses = {"not_started": StepStatus.NOT_STARTED,
                          "in_progress": StepStatus.IN_PROGRESS,
                          "completed": StepStatus.COMPLETED}

        try:
            # Update each step's status
            for idx, status_str in enumerate(statuses):
                # Convert string status to enum
                if status_str not in valid_statuses:
                    raise ValueError(f"Invalid status: {status_str}")

                # Transition the step to the new status
                new_status = valid_statuses[status_str]
                self.plan.steps[idx].transition(new_status)

        except ValueError as e:
            return f"Update status error: {str(e)}"

        return f"Plan status updated: \n {str(self.plan)}"
