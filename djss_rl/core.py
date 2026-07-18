"""Extracted DJSS RL implementation from the cleaned notebook.

This module intentionally preserves the notebook behavior while making the
environment, agent, training, and evaluation code importable from tests and
terminal commands.
"""
# Runtime defaults matching the safe notebook path.
validation = False
maintenance_integrated = False


# ---- Extracted notebook cell 12 ----
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "djss-rl-matplotlib"))

import gymnasium as gym
from gymnasium import spaces
Env = gym.Env
import numpy as np
import math
import re
from collections import deque
import pandas as pd
import random
import datetime
import plotly.figure_factory as ff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.distributions import Categorical
import plotly.graph_objects as go
import copy
import wandb
import optuna
from configparser import ConfigParser
import ast
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import weibull_min
import pickle
import warnings
# Create a SummaryWriter
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---- Extracted notebook cell 14 ----
class OperationSelector:
    def __init__(self, world):
        # Initialize the OperationSelector with a reference to the world object
        self.world = world

    def select_operation(self, machine, heuristic):
        # Apply the heuristic function to the machine's legal actions to select an operation
        operation = heuristic(machine.legal_actions)
        return operation

    def MRT_DR_O(self, machine):
        """Select the operation with the most remaining processing time (MRT) for the current machine."""
        self.world.MRT_DR_O += 1  # Increment the counter for MRT
        return self.select_operation(machine, lambda operation: max(operation, key=lambda op: op[0].parent.remaining_processing_time)[0])

    def LRT_DR_O(self, machine):
        """Select the operation with the least remaining processing time (LRT) for the current machine."""
        self.world.LRT_DR_O += 1  # Increment the counter for LRT
        return self.select_operation(machine, lambda operation: min(operation, key=lambda op: op[0].parent.remaining_processing_time)[0])

    def SPT_DR_O(self, machine):
        """Select the operation with the shortest processing time (SPT) for the current machine."""
        self.world.SPT_DR_O += 1  # Increment the counter for SPT
        return self.select_operation(machine, lambda operation: min(operation, key=lambda op: op[1])[0])

    def EDD_DR_O(self, machine):
        """Select the operation with the earliest due date (EDD) for the current machine."""
        self.world.EDD_DR_O += 1  # Increment the counter for EDD
        return self.select_operation(machine, lambda operation: min(operation, key=lambda op: op[0].parent.due_date)[0])

    def MCR_DR_O(self, machine):
        """Select the operation with the minimal Critical Ratio (MCR) for the current machine (the whole job)."""
        self.world.MCR_DR_O += 1  # Increment the counter for MCR
        return self.select_operation(machine, lambda operation: min(operation, key=lambda op: (op[0].parent.due_date - self.world.total_timestamp) / op[0].parent.remaining_processing_time)[0])

    def CR_DR_O(self, machine):
        """Select the operation with the smallest Critical Ratio (CR) for the current machine (the current operation)."""
        self.world.CR_DR_O += 1  # Increment the counter for CR
        return self.select_operation(machine, lambda operation: min(operation, key=lambda op: (op[0].parent.due_date - self.world.total_timestamp) / op[1])[0])

    def SLK_DR_O(self, machine):
        """Select the operation with the smallest Slack Time (SLK) for the current machine."""
        self.world.SLK_DR_O += 1  # Increment the counter for SLK
        return self.select_operation(machine, lambda operation: min(operation, key=lambda op: (op[0].parent.due_date - self.world.total_timestamp - op[0].parent.remaining_processing_time))[0])

    def ATC_DR_O(self, machine):
        """Select the operation with the highest Apparent Tardiness Cost (ATC) for the current machine."""
        self.world.ATC_DR_O += 1  # Increment the counter for ATC
        k = 0.5  # Scaling parameter (can be tuned)
        return self.select_operation(machine, lambda operation: max(operation, key=lambda op: (np.exp(-(op[0].parent.due_date - self.world.total_timestamp - op[0].parent.remaining_processing_time) / (k * op[1]))) / op[1])[0])

    def LSPO_DR_O(self, machine):
        """Select the operation with the least Slack per Operation (LSPO) for the current machine."""
        self.world.LSPO_DR_O += 1  # Increment the counter for LSPO
        return self.select_operation(machine, lambda operation: min(operation, key=lambda op: (op[0].parent.due_date - self.world.total_timestamp - op[0].parent.remaining_processing_time) / len(op[0].parent.operations))[0])


# ---- Extracted notebook cell 16 ----
def failure_risk(machine):
  """Calculate the failure risk of a machine based on its virtual age (i.e., the cumulative processing time from the last maintenance task)."""
  # Compute the failure risk using the Weibull cumulative distribution function (CDF)
  return 1 - np.exp(-(machine.virtual_age/machine.scale)**machine.shape)


# ---- Extracted notebook cell 18 ----
class Job_Shop_Env(Env):
    # Initialization method
    def __init__(self, world, reset_callback=None, reward_callback=None, observation_callback=None, info_callback=None, done_callback=None):
        super(Job_Shop_Env, self).__init__() # Initialize the base Env class
        self.world = world # Reference to the world object containing the environment's state
        self.operations_done = world.operations_done # List of completed operations
        self.machines = world.machines # List of machines in the environment
        self.jobs = world.jobs  # List of jobs to be processed
        self.reward = 0 # Initialize the reward variable

        # scenario callbacks
        self.reset_callback = reset_callback # Function to reset the environment
        self.reward_callback = reward_callback # Function to compute the reward
        self.observation_callback = observation_callback # Function to get the observation
        self.info_callback = info_callback # Additional info callback (not used)
        self.done_callback = done_callback # Function to check if the episode is done

        self.chosen_operation = OperationSelector(world) # Initialize the operation selector

        # Variables for plotting and visualization
        self.global_actions = [] # List to store all actions for Gantt chart

        # Generate random colors for plotting, one for each job plus extra
        self.colors = [tuple([random.random() for _ in range(3)]) for _ in range(len(self.jobs) + 2)]

        # Configure the observation and action spaces
        self.configure_spaces()

        # Define the list of operation sequencing actions (heuristics)
        self.action_mapping_operation_sequencing_list = [ # Action set for job sequencing
            lambda machine: self.chosen_operation.MRT_DR_O(machine),  # Most Remaining Processing Time
            lambda machine: self.chosen_operation.LRT_DR_O(machine),  # Least Remaining Processing Time
            lambda machine: self.chosen_operation.SPT_DR_O(machine),  # Shortest Processing Time
            lambda machine: self.chosen_operation.EDD_DR_O(machine),  # Earliest Due Date
            lambda machine: self.chosen_operation.MCR_DR_O(machine),  # Minimal Critical Ratio
            lambda machine: self.chosen_operation.CR_DR_O(machine),   # Critical Ratio
            lambda machine: self.chosen_operation.ATC_DR_O(machine),  # Apparent Tardiness Cost
            lambda machine: self.chosen_operation.LSPO_DR_O(machine), # Least Slack per Operation
            lambda machine: self.chosen_operation.SLK_DR_O(machine),  # Smallest Slack Time
        ]

    # Function to configure observation and action spaces
    def configure_spaces(self):
      # Get the initial state using the observation callback
      state = self.observation_callback(self.world)
      # Define the observation space based on the state shape
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=state.shape, dtype=np.float32)
      # Define the action space as discrete, number of actions depends on maintenance integration
      actions = 9 + int(maintenance_integrated)
      self.action_space = spaces.Discrete(actions)

    # Function to get legal actions for a specific machine
    def get_legal_actions(self, machine): # Return the candidate operations of a specific machine
      # Set of completed operations (job, order)
      operations_done_set_info = {(op.parent, op.order) for op in self.world.operations_done}
      # Filter operations that can legally be processed on the machine
      machine.legal_actions = [operation for operation in machine.operations_to_process if operation[0].parent.legal and operation[0] not in self.world.operations_done and ((operation[0].parent, operation[0].order - 1) in operations_done_set_info or operation[0].order==0)]

    # A function to execute  preventive maintenance once scheduled
    def execute_PM(self, machine):
      machine.total_maintenance_cost += machine.PM_cost # Add PM cost to total maintenance cost
      machine.latest_PM = self.world.total_timestamp # Update the timestamp of the latest PM
      machine.PM_actions_counts += 1 # Increment the count of PM actions
      machine.threshold = machine.initial_threshold + machine.PM_duration + self.world.total_timestamp
      machine.virtual_age = 0 # Reset the virtual age of the machine
      machine.action = ["Predictive Maintenance", machine.PM_duration, self.world.total_timestamp * 60]
      # Add the maintenance action to the global actions list for visualization
      self.global_actions.extend([[machine.name] + machine.action])
      machine.request = False # Reset the machine's request flag

    # Function to execute an operation on a machine
    def execute_operation(self, machine, operation):
      machine.assigned_op += 1 # Increment the count of assigned operations
      operation_parent = operation.parent # Get the job to which the operation belongs
      operation_parent.legal = False # Mark the job as illegal to prevent re-scheduling
      # Set the machine's current action with the operation, its processing time, and start time
      machine.action = [operation, operation.available_machines[machine], self.world.total_timestamp * 60]
      machine.workload += operation.available_machines[machine] # Update machine's workload
      # Update total energy consumption based on operation processing time
      machine.total_energy_consumption += machine.energy_consumption * operation.available_machines[machine]
      # Check if the job becomes tardy due to this operation
      if operation_parent.tardy_operations == 0 and self.world.total_timestamp >= operation_parent.due_date and operation_parent.remaining_processing_time > 0: # Track the actual tardiness of the job that includes the current operation
        operation_parent.tardy_operations = len(operation_parent.operations) - operation_parent.op_processed
      # Update the job's remaining processing time
      operation_parent.remaining_processing_time -= int(operation.average_processing_time) # Update the remaining processing time of the job
      operation_parent.op_processed += 1 # Increment the count of processed operations
      operation_parent.CRJ = operation_parent.op_processed / len(operation_parent.operations) # Update the completion rate of the job
      self.global_actions.extend([[machine.name] + machine.action]) # Update the list containing all the actions
      machine.request = False # Reset the machine's request flag

    # Function to update the environment's state by advancing to the next event
    def update_state(self):
      # Find and calculate the closest job arrival, the minimum processing time among machines, and the closest failure
      closest_arrival_time = min((job.arrival_time - self.world.total_timestamp for job in self.jobs if job.arrival_time > self.world.total_timestamp), default=500000) # Large default value if no arrivals
      # Find the minimum remaining processing time among all machines
      min_PT = min((machine.action[1] for machine in self.machines if machine.action[1] > 0), default=500000) # Large default value if no processing time
      # Calculate the closest machine failure time
      closest_failure = min((machine.threshold - self.world.total_timestamp for machine in self.machines if machine.action != [0, 0, 0]), default=500000) # Large default value if no failures
      # List of machines that will fail at the closest failure time
      min_threshold_elements = [machine for machine in self.machines if machine.threshold - self.world.total_timestamp == closest_failure]

      # determine and Compute the minimum of the key events: operation completion, job insertion, and machine failure
      min_event = min(closest_arrival_time, min_PT, closest_failure)

       #The following code does consider 7 scenarios : a combination of the three events (operation completion, new job insertion, and machine breakdown)
      if min(closest_arrival_time, min_PT) > closest_failure : # Close Machine(s) Failure
        self.handle_case1(closest_failure, min_threshold_elements)
      elif min_PT == closest_failure and min_PT < closest_arrival_time: # Both Failure and operation completion happen simultanously
        self.handle_case2(closest_failure, min_threshold_elements)
      elif closest_arrival_time == closest_failure and min_PT > closest_arrival_time: # Both Failure and job insertion happen simultanously
        self.handle_case3(closest_failure, min_threshold_elements)
      elif closest_arrival_time == closest_failure == min_PT and min_PT != 500000: # All the events happen simultanously
        self.handle_case4(closest_failure, min_threshold_elements)
      elif closest_arrival_time < min(min_PT, closest_failure): # Job insertion
        self.handle_case5(closest_arrival_time)
      elif min_PT == closest_arrival_time and min_PT < closest_failure: # Both job insertion and operation completion happen simultanously
        self.handle_case6(min_PT)
      elif min(closest_arrival_time, closest_failure) > min_PT:
        self.handle_case7(min_PT)  # Operation completion

    # Function to process an action from the agent
    def step(self, action, **kwargs):
      machine = self.world.ready_machine # Get the idle machine that requests processing at the current decision time

      operation_sequencing = action  # The action corresponds to an index in the action mapping list

      # Handle operation sequencing and maintenance if integrated
      if maintenance_integrated and operation_sequencing == len(self.action_mapping_operation_sequencing_list):
        self.execute_PM(machine)
      else:
        # Select the operation to process based on the chosen heuristic
        operation_to_process = self.action_mapping_operation_sequencing_list[operation_sequencing](machine)
        self.execute_operation(machine, operation_to_process)

      self.reward = self._get_reward  # Get the reward using the reward callback

      obs_n = self._get_obs #Get the new global(observation) state

      #If there are no other processing requests, move the shop floor forward min_PT steps
      while True:
        # Check if there is request from any machine or if jobs are completed
        if all(job.CRJ==1 for job in self.jobs) or (any(machine.request for machine in self.machines) and any(job.legal for job in self.jobs)):
          break
        self.update_state() # Update the environment's state

      done = all(job.CRJ==1 for job in self.jobs) # Check if the episode is done

      return obs_n, self.reward, done, False, {}

    # A function to handle job arrivals
    def handle_job_arrivals(self): #The new job arrival process is considered by adding the newly arrived job to the waiting buffer.
      new_jobs = [job for job in self.jobs if job.arrival_time == self.world.total_timestamp] # Get the list of new jobs arriving at the current timestamp
      self.world.new_job.extend(new_jobs) # Add new jobs to the world's list
      for job in new_jobs:
        job.legal = job.inserted = True # Mark the job as legal and inserted
      for machine in self.machines:
        # Update machine's action count based on new operations available
        machine.actions += sum(1 for job in new_jobs for op in job.operations if machine in op.available_machines)

    # Function to handle different cases of events (failure, arrival, completion)
    def handle_case1(self, closest_failure, min_threshold_elements):
      #Case when there is a failure to occur
      self.world.total_timestamp += closest_failure  # Advance the timestamp
      self.handle_corrective_maintenance(min_threshold_elements, closest_failure) # Handle maintenance

    def handle_case2(self, closest_failure, min_threshold_elements):
      # Case when failure and operation completion happen simultaneously
      self.world.total_timestamp += closest_failure # Advance the timestamp
      # Machines that complete processing at this time
      machines_min_PT = [machine for machine in self.machines if machine.action[1] == closest_failure]
      self.process_ready_machines(machines_min_PT, closest_failure)  # Process these machines
      self.handle_corrective_maintenance(min_threshold_elements, closest_failure, machines_min_PT) # Maintenance

    def handle_case3(self, closest_failure, min_threshold_elements):
      # Case when failure and job insertion happen simultaneously
      self.world.total_timestamp += closest_failure # Advance the timestamp
      self.handle_job_arrivals() # Handle job arrivals
      self.handle_corrective_maintenance(min_threshold_elements, closest_failure) # Maintenance

    def handle_case4(self, closest_failure, min_threshold_elements):
      #Case when the three scenarios happen simultanously
      self.world.total_timestamp += closest_failure  # Advance the timestamp
      self.handle_job_arrivals() # Handle job arrivals
      machines_min_PT = [machine for machine in self.machines if machine.action[1] == closest_failure]
      self.process_ready_machines(machines_min_PT, closest_failure) # Process these machines
      self.handle_corrective_maintenance(min_threshold_elements, closest_failure, machines_min_PT) # Maintenance

    def handle_case5(self, closest_arrival_time):
      #Case of a new job insertion only
      self.world.total_timestamp += closest_arrival_time  # Advance the timestamp
      self.handle_job_arrivals()# Handle job arrivals
      self.process_other_machines(self.machines, closest_arrival_time) # Update other machines

    def handle_case6(self, min_PT):
      #Case of job insertion and a machine completes the in-process operation simultaneousely
      self.world.total_timestamp += min_PT # Advance the timestamp
      self.handle_job_arrivals()  # Handle job arrivals
      machines_min_PT = [machine for machine in self.machines if machine.action[1] == min_PT]
      self.process_ready_machines(machines_min_PT, min_PT) # Process these machines
      other_machines = [machine for machine in self.machines if machine not in machines_min_PT]
      self.process_other_machines(other_machines, min_PT)  # Update other machines

    def handle_case7(self, min_PT):
      #Case when only a machine completes processing
      self.world.total_timestamp += min_PT # Advance the timestamp
      # Machines that complete processing at this time
      machines_min_PT = [machine for machine in self.machines if machine.action[1] == min_PT]
      self.process_ready_machines(machines_min_PT, min_PT) # Process these machines
      other_machines = [machine for machine in self.machines if machine not in machines_min_PT]
      self.process_other_machines(other_machines, min_PT) # Update other machines

    # A function to update machines that finish the in-process operation
    def process_ready_machines(self, machines_min_PT, min_PT):
      for machine in machines_min_PT :
        action = machine.action[0]
        if action not in {"Predictive Maintenance", "Corrective Maintenance"}:
          self.world.operations_done.append(action)  # Add the completed operation to the list of processed operations
          machine.utilization = machine.workload / self.world.total_timestamp  # Update utilization
          machine.virtual_age += min_PT  # Update virtual age
          parent = action.parent # Get the job
          if action.order == len(parent.operations) - 1:
            parent.finish_processing = self.world.total_timestamp # Mark job as finished
          parent.legal = parent.op_assigned != len(parent.operations) # Update job's legality
        self.get_legal_actions(machine) #Get new legal actions/operations for the machine
        machine.action = [0, 0, 0]  # Reset the machine's action
        machine.request = any(machine.legal_actions)  # Update the request flag

    # A function to update the other machines
    def process_other_machines(self, other_machines, x):
      for machine in other_machines:
        machine.action[1] = max(machine.action[1] - x, 0) #All busy machines move x steps # Reduce remaining processing time
        if all(y==0 for y in machine.action): # If machine is idle
          machine.waiting_time += x # Increase waiting time
          machine.threshold += x # Increase failure threshold
          self.get_legal_actions(machine) #Get new legal actions/operations
          machine.request = any(machine.legal_actions) # Update request flag
          # Update idle energy consumption
          machine.total_energy_consumption += machine.idle_energy_consumption * x
        if machine.action[0] not in {0, "Predictive Maintenance", "Corrective Maintenance"}:
          machine.virtual_age += x # Update virtual age
        machine.utilization = machine.workload / self.world.total_timestamp # Update utilization

    # A function to handle corrective maintenance
    def handle_corrective_maintenance(self, min_threshold_elements, closest_failure, machines_min_PT=[]):
      other_machines = [machine for machine in self.machines if machine not in machines_min_PT]

      failed_machines = min_threshold_elements # Machines that have failed

      for machine in failed_machines:
        if any(machine.action) and machine.action[1] != 0 and not isinstance(machine.action[0], str):
          # If machine was processing an operation, handle the interruption
          action_parent = machine.action[0].parent # Get the job
          action_parent.legal = True  # Mark job as legal again
          action_parent.op_assigned -= 1 # Decrement assigned operations
          action_parent.op_processed -= 1 # Decrement processed operations
          action_parent.CRJ = action_parent.op_processed / len(action_parent.operations)  # Update completion rate
          action_parent.remaining_processing_time += int(machine.action[0].average_processing_time) # Update remaining time
          machine.workload -= machine.action[1] # Update workload
          # Adjust energy consumption
          machine.total_energy_consumption -= machine.energy_consumption * machine.action[1]
          machine.utilization = machine.workload / self.world.total_timestamp # Update utilization
          # Update the action's finish time in the global actions list
          op_index = next((i for i, sublist in enumerate(self.global_actions) if sublist[1] == machine.action[0]))
          self.global_actions[op_index][2] = self.world.total_timestamp - (self.global_actions[op_index][3] / 60)

        machine.total_maintenance_cost += machine.failure_cost  # Add failure cost
        machine.CM_actions_counts += 1 # Increment corrective maintenance count
        timestep = machine.threshold  # Get the failure time
        machine.threshold = machine.initial_threshold + machine.CM_duration + self.world.total_timestamp
        machine.virtual_age = 0 # Reset virtual age
        # Set the machine's action to corrective maintenance
        machine.action = ["Corrective Maintenance", machine.CM_duration, timestep * 60]

        self.global_actions.extend([[machine.name] + machine.action]) # Add to global actions

        if machine in other_machines:
          other_machines.remove(machine) # Remove from other machines

      for machine in machines_min_PT:
        self.get_legal_actions(machine) # Get legal actions for machines that finished processing
        machine.request = any(machine.legal_actions)  # Update request flag

      self.process_other_machines(other_machines, closest_failure)  # Update other machines

    # Property to get the makespan (total completion time)
    @property
    def _get_makespan(self): # Get the maximal completion time of the last operation
      return self.world.total_timestamp # Return the current total timestamp

    # Function to reset the environment
    def reset(self, *, seed=None, options=None):
      super().reset(seed=seed)
      self.global_actions = [] # Clear global actions
      if self.reset_callback:
        self.reset_callback(self.world)
      return self._get_obs, {}

    # Property to get the current observation
    @property
    def _get_obs(self):
      # Get the observation using the observation callback
      return self.observation_callback(self.world) if self.observation_callback is not None else np.zeros(0)

    # Property to get the current reward
    @property
    def _get_reward(self):
      return self.reward_callback(self.world) if self.reward_callback else 0.0

    # Function to render the environment (e.g., create a Gantt chart)
    def render(self, mode='human'):
      # Prepare data for the Gantt chart
      df = [{
        "Task": str(action[0]),
        "Start": datetime.datetime.fromtimestamp(action[3]).strftime('%Y-%m-%d %H:%M:%S'),  # Format the timestamp as needed
        "Finish": datetime.datetime.fromtimestamp(action[3] + action[2] * 60).strftime('%Y-%m-%d %H:%M:%S'),
        "Resource": str(action[1].parent.name) if action[1] not in {"Corrective Maintenance", "Predictive Maintenance"} else str(action[1])
      } for action in self.global_actions]

      fig = ff.create_gantt(df, index_col='Resource', colors=self.colors, show_colorbar=True, group_tasks=True, bar_width=0.25)

      # Update the chart's appearance
      fig.update_yaxes(autorange="reversed", tickfont=dict(size=20, family='Arial', color='black'))
      fig.update_xaxes(tickfont=dict(size=20, family='Arial', color='black'))
      fig.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size=20), traceorder='normal', bordercolor='black', borderwidth=2, orientation="v"),
                        autosize=False, width=1500, height=600)
      return fig


# ---- Extracted notebook cell 20 ----
def make_env(dataset_path=None):
    scenario = Scenario()
    world = scenario.make_world(dataset_path=dataset_path)
    env = Job_Shop_Env(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
    )
    return env


# ---- Extracted notebook cell 22 ----
# Define the Operation class
class Operation(object):
    # Use __slots__ to declare data members and optimize memory usage
    __slots__ = ['name', 'id', 'parent', 'order', 'available_machines', 'average_processing_time']

    # Initialization method
    def __init__(self):
      # Name of the operation (string)
      self.name = ''
      # Reference to the parent job (Job object)
      self.parent = None
      # Sequence order of the operation within the job (integer)
      self.order = None
      # List of machines that can process this operation (list of Machine objects)
      self.available_machines = []
      # Average processing time for the operation (float)
      self.average_processing_time = 0
      # Note: 'id' is declared in __slots__ but not initialized here; consider initializing if needed

# Define the Job class
class Job(object):
    # Declare data members using __slots__
    __slots__ = ['name', 'id', 'operations', 'arrival_time', 'due_date', 'op_assigned', 'op_processed', 'processing_time', 'remaining_processing_time', 'tardy_operations',
                 'CRJ', 'finish_processing', 'tardiness_cost', 'legal', 'inserted', 'cancelled']

    # Initialization method
    def __init__(self):
      # Name of the job (string)
      self.name = ''
      # List of operations belonging to the job (list of Operation objects)
      self.operations = []
      # Arrival time of the job into the system (float or integer)
      self.arrival_time = self.due_date = self.op_assigned = self.processing_time = 0 # Due date of the job (float or integer)  # Number of operations assigned to machines (integer) # Total processing time of the job (float)
      self.remaining_processing_time = self.tardy_operations = self.CRJ = self.finish_processing = 0 # Remaining processing time for the job (float)  # Number of tardy operations in the job (integer) # Completion rate of the job (float between 0 and 1)  # Timestamp when the job finishes processing (float)
      self.tardiness_cost = 0 # Cost associated with tardiness of the job (float)
      self.legal = True # Indicates if the job is eligible for scheduling (boolean)
      self.inserted = self.cancelled = False # Indicates if the job has been inserted into the system (boolean)  # Indicates if the job has been cancelled (boolean)
      # Note: 'id' is declared in __slots__ but not initialized here; consider initializing if needed

# Define the Work_Center class
class Work_Center(object):
  # Declare data members using __slots__
  __slots__ = ['name', 'id', 'availability', 'machines', 'waiting_time', 'utilization', 'workload', 'assigned_op']

  # Initialization method
  def __init__(self):
    # Name of the work center (string)
    self.name = ''
    # List of machines within the work center (list of Machine objects)
    self.machines = []
    # Availability status of the work center (integer or boolean)
    self.availability = self.waiting_time = self.utilization = self.workload = self.assigned_op = 0 # Total waiting time at the work center (float) # Utilization rate of the work center (float between 0 and 1) # Total workload assigned to the work center (float) # Number of operations assigned to the work center (integer)
    # Note: 'id' is declared in __slots__ but not initialized here; consider initializing if needed

# Define the Machine class
class Machine(object):
  # Declare data members using __slots__
  __slots__ = ['name', 'id', 'operations_to_process', 'legal_actions', 'request', 'buffer', 'action', 'actions', 'availability', 'total_maintenance_cost', 'latest_PM',
               'scale', 'shape', 'threshold', 'initial_threshold', 'PM_actions_counts', 'CM_actions_counts', 'PM_duration', 'virtual_age', 'assigned_op'
                'total_maintenance_time', 'total_energy_consumption', 'energy_consumption', 'idle_energy_consumption', 'waiting_time', 'utilization', 'buffer_time', 'workload', 'assigned_op', 'failure_cost', 'CM_duration', 'PM_cost']

  # Initialization method
  def __init__(self):
    # Name of the machine (string)
    self.name = ''
    # List of operations that can be processed by the machine (list of Operation objects)  # List of legal actions (operations that can currently be assigned) (list of Operation objects)
    self.operations_to_process = self.legal_actions = []
    # Current action of the machine: [operation, processing time, start time] or maintenance action (list)
    self.action = [0, 0, 0]
    # Flag indicating if the machine is requesting an operation (boolean)
    self.request = True
    # Availability status of the machine (integer or boolean)  # Total maintenance cost accumulated by the machine (float) # Timestamp of the latest preventive maintenance performed (float)
    self.availability = self.total_maintenance_cost = self.PM_duration =  0
    self.waiting_time = self.CM_duration = self.failure_cost = self.PM_cost = self.utilization = self.workload = self.assigned_op = 0

# Define the World class representing the simulation environment
class World(object):
  # Declare data members using __slots__
  __slots__ = ['DDT', '_lambda', 'work_centers', 'machines', 'operations', 'jobs', 'operations_done', 'total_operations', 'ready_machine', 'ready_operation', 'new_job', 'total_cost', 'previous_tardiness_rate',
               'total_timestamp', 'total_tardiness_cost', 'tardiness_rate', 'total_energy_consumption', 'total_maintenance_cost', 'start', 'MRT_DR_O', 'LRT_DR_O', 'SPT_DR_O', 'EDD_DR_O', 'MCR_DR_O',
               'SLK_DR_O', 'CR_DR_O', 'ATC_DR_O', 'LSPO_DR_O', '_lambda']

  # Initialization method
  def __init__(self):
    # List of new jobs arriving into the system (list of Job objects)
    self.new_job = None
    # List of work centers in the environment (list of Work_Center objects)
    self.work_centers = []
    # List of machines in the environment (list of Machine objects)
    self.machines = []
    # List of all operations in the environment (list of Operation objects)
    self.operations = []
    # List of jobs in the environment (list of Job objects)
    self.jobs = []
    # List of operations that have been completed (list of Operation objects)
    self.operations_done = []
    # Machine ready to process an operation (Machine object) # Operation ready to be processed (Operation object)
    self.ready_machine = self.ready_operation = None


# ---- Extracted notebook cell 24 ----
# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()


# ---- Extracted notebook cell 26 ----
#This scenario will create many data sets with one disturbance : job insertion
class Scenario(BaseScenario):
  # Method to create the simulation world
  def make_world(self, dataset_path=None):
      if dataset_path:
        return self.make_world_from_dataset(dataset_path)

      world = World()  #Create a new World object representing the simulation environment

      # add work centers
      m = 5 #random.randint(4,7) # Number of work centers (could also be randomized)
      world.DDT = 0.5#random.choice([0.5,1,1.5]) # Due date tightness , randomly chosen
      world.work_centers = [Work_Center() for _ in range(m)] # Create a list of Work_Center objects
      p = 0  # Counter for assigning unique machine IDs
      for i, work_center in enumerate(world.work_centers, 1):
        work_center.id = i - 1 # Assign an ID to the work center
        work_center.name = 'Work Center %d' % i # Name the work center
        machines_per_work_center = 3 # Number of machines in each work center
        work_center.machines = [Machine() for _ in range(machines_per_work_center)] # Create Machine objects

        # Initialize each machine in the work center
        for k, machine in enumerate(work_center.machines, 1):
          machine.id = p # Assign a unique ID to the machine
          machine.name = 'Machine %d,%d' % (i,k) # Name the machine
          #Weibull distriubtion
          machine.shape = round(random.uniform(2.1, 2.7), 1) #defines the shape of the failure rate curve over time (we consider an increasing pattern in our case)
          machine.scale = random.randint(20, 40) * 60  #average time to failure of the distribution in minutes
          # Calculate the expected lifetime (mean time to failure) using the Weibull distribution
          machine.threshold = machine.initial_threshold = int(weibull_min.ppf(0.9, machine.shape, loc=0, scale=machine.scale))
          # Corrective maintenance action
          machine.failure_cost = random.randint(8000,20000) # Cost of corrective maintenance
          machine.CM_duration = random.randint(480,620)  # Duration of corrective maintenance in minutes
          # Preventive maintenance parameters
          machine.PM_cost = int(machine.failure_cost/3)
          machine.PM_duration = int(machine.CM_duration/3)
          machine.energy_consumption = round(random.uniform(0.2, 0.9), 2) # During operation ($ per minute)
          machine.idle_energy_consumption = round(random.uniform(0.01, 0.02), 2) # When idle ($ per minute)
          p += 1 # Increment the machine ID counter

      world.machines = [machine for work_center in world.work_centers for machine in work_center.machines]

      # add jobs and operations
      available_machines_per_operation=[]
      n = 50# random.choice([25,50,75]) # Number of jobs (could also be randomized)
      world.jobs = [Job() for _ in range(n)] # Create Job objects
      p = 0 # Counter for assigning unique operation IDs
      for i, job in enumerate(world.jobs,1):
          job.name = 'Job %d' % i # Name the job
          job.id = i - 1 # Assign an ID to the job
          op = random.randint(6,10) # Tasks # Number of operations in the job
          job.operations = [Operation() for _ in range(op)] # Create Operation objects

          # Initialize each operation in the job
          for k, operation in enumerate(job.operations,1):
            operation.id = p  # Assign a unique ID to the operation
            p += 1 # Increment the operation ID counter
            operation.name = 'Operation %d,%d' % (i,k) # Name the operation
            operation.parent = job # Set the parent job of the operation
            operation.order = k - 1 # Sequence order of the operation within the job

            # Assign available machines for the operation
            available_machines = random.sample(world.machines, random.randint(1, m*3)) # An operation could be processed by one or multiple machines
            # Assign processing times on each available machine
            operation.available_machines = {machine: random.randint(60, 120) for machine in available_machines} # procesing time in minutes
            operation.average_processing_time = np.mean(list(operation.available_machines.values())) # Average processing time
            # Add the operation to each machine's list of operations to process
            for agent, time in operation.available_machines.items():
              agent.operations_to_process.extend([(operation, time)])

          # Calculate the total processing time of the job
          job.processing_time = sum([int(operation.average_processing_time) for operation in job.operations])
          job.remaining_processing_time = job.processing_time # Initialize remaining processing time
          # Assign a tardiness cost per minute for the job
          job.tardiness_cost = round(random.uniform(0.3,1.3), 1)

      # Calculate the total number of operations in the world
      world.operations = sum([len(job.operations) for job in world.jobs])

      # Assign arrival times to jobs
      # Randomly select jobs that are available at time t = 0
      random_initial_jobs = [job.name for job in random.sample(world.jobs, 25)]
      job_dict = {job.name: job for job in world.jobs} # Create a dictionary to map job names to job objects
      random_initial_jobs = [job_dict[x] for x in random_initial_jobs] # Get the Job objects for initial jobs
      print("{} jobs are available at time t = 0".format(len(random_initial_jobs))) # Print the number of initial jobs

      # Set arrival times and due dates for initial jobs
      for job in random_initial_jobs:
        job.arrival_time = 0 # Arrival time at t = 0
        #Due data = arrival_time + sum(processing time) * DDT
        job.due_date = int(job.arrival_time + job.processing_time * world.DDT) #min

      # Assign arrival times to the remaining jobs using an exponential distribution
      world._lambda = 50#random.choice([50, 100, 200]) # Job arrival rate parameter
      x= 1 / world._lambda    # Mean inter-arrival time  # A new job (on average) is inseterd each 50,100, or 200 minutes
      _arrival_time = 0  # Initialize the arrival time accumulator

      # Iterate over jobs not initially available to assign arrival times and due dates
      for job in list(set(world.jobs) - set(random_initial_jobs)):
          #Get the next probability value from Uniform(0,1)
          p = random.random() # Generate a random probability from Uniform(0,1)
          #Plug it into the inverse of the CDF of Exponential(_lambda)
          _inter_arrival_time = -math.log(1.0 - p)/x # Compute inter-arrival time using inverse transform sampling
          #Add the inter-arrival time to the running sum
          _arrival_time += _inter_arrival_time # Update the arrival time
          job.arrival_time = int(_arrival_time)  #Assign the arrival time to the job
          #Due data = arrival_time + sum(processing time) * DDT
          job.due_date = int(job.arrival_time + job.processing_time * world.DDT) #min


      if validation: # Store all attribute values for future reuse
        config = ConfigParser() # Create a ConfigParser object
        config.add_section('world') # Add a section named 'world'
        config.set('world', 'DDT', str(world.DDT))
        config.set('world', 'number of jobs', str(n))
        config.set('world', 'job arrival rate', str(world._lambda))
        for i, machine in enumerate(world.machines, 1):
          config.set('world', 'scale parameter´s value of machine {}'.format(i), str(machine.scale))
          config.set('world', 'shape parameter´s value of machine {}'.format(i), str(machine.shape))
          config.set('world', 'failure cost of machine {}'.format(i), str(machine.failure_cost))
          config.set('world', 'corrective maintenance duration of machine {}'.format(i), str(machine.CM_duration))
          config.set('world', 'PM cost of machine {}'.format(i), str(machine.PM_cost))
          config.set('world', 'PM duration of machine {}'.format(i), str(machine.PM_duration))
          config.set('world', 'energy consumption of machine {}'.format(i), str(machine.energy_consumption))
          config.set('world', 'idle energy consumption of machine {}'.format(i), str(machine.idle_energy_consumption))
        for i, job in enumerate(world.jobs, 1):
          config.set('world', 'operations of job {}'.format(i), str(len(job.operations)))
          config.set('world', 'Unit tardiness cost of job {}'.format(i), str(job.tardiness_cost))
          config.set('world', 'arrival time of {}'.format(job.name), str(job.arrival_time))
          for k, operation in enumerate(job.operations, 1):
            config.set('world', 'operation {} of job {} available machines'.format(k,i), str(operation.available_machines))

        with open('Dataset {}_{}_{}.ini'.format(n, world.DDT, x), 'w') as conf:
          config.write(conf)

      # make initial conditions
      self.reset_world(world)
      print("DDT :",world.DDT, "lambda :", world._lambda, "there are {} jobs".format(len(world.jobs)),"and {} machines".format(len(world.machines)), "and {} operations".format(sum([len(job.operations) for job in world.jobs])), "and {} minutes".format(sum([int(operation.average_processing_time) for job in world.jobs for operation in job.operations])))
      return world

  def _dataset_get(self, section, key, cast=str):
      value = section[key]
      return cast(value) if cast is not str else value

  def _parse_available_machine_tokens(self, value):
      pattern = r'<__main__\.Machine object at (0x[0-9a-fA-F]+)>:\s*(\d+)'
      return [(match.group(1), int(match.group(2))) for match in re.finditer(pattern, value)]

  def _parse_available_machine_entries(self, value):
      legacy_entries = self._parse_available_machine_tokens(value)
      if legacy_entries:
        return legacy_entries

      try:
        parsed_value = ast.literal_eval(value)
      except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Unable to parse available machines value: {value}") from exc

      if isinstance(parsed_value, dict):
        raw_entries = parsed_value.items()
      elif isinstance(parsed_value, (list, tuple)):
        raw_entries = parsed_value
      else:
        raise ValueError(f"Available machines must be a dict or list of pairs: {value}")

      entries = []
      for machine_ref, processing_time in raw_entries:
        if isinstance(machine_ref, int):
          machine_token = f"machine:{machine_ref}"
        else:
          machine_text = str(machine_ref).strip()
          machine_match = re.search(r"(\d+)$", machine_text)
          if not machine_match:
            raise ValueError(f"Machine reference does not contain an ID: {machine_ref}")
          machine_token = f"machine:{int(machine_match.group(1))}"
        entries.append((machine_token, int(processing_time)))
      return entries

  def make_world_from_dataset(self, dataset_path, machine_address_strategy="first_seen"):
      """Build a World from a dataset .ini generated by this notebook.

      The original dataset stores operation-compatible machines as Python object
      memory addresses, not machine IDs. The loader maps those address tokens
      deterministically to the restored machine list. This preserves operation
      processing times and routing cardinality, but the exact original machine
      identity cannot be guaranteed unless the dataset is re-exported with
      machine IDs or names.
      """
      config = ConfigParser()
      read_files = config.read(dataset_path, encoding="utf-8")
      if not read_files:
        raise FileNotFoundError(f"Dataset file not found or unreadable: {dataset_path}")
      if not config.has_section('world'):
        raise ValueError(f"Dataset file has no [world] section: {dataset_path}")

      data = config['world']
      world = World()
      world.DDT = float(data['ddt'])
      world._lambda = int(float(data['job arrival rate']))

      machine_ids = sorted(
        int(match.group(1))
        for key in data.keys()
        for match in [re.match(r"scale parameter.+ machine (\d+)$", key)]
        if match
      )
      if not machine_ids:
        raise ValueError("Dataset does not contain machine parameter entries.")

      machines_per_work_center = 3
      work_center_count = math.ceil(len(machine_ids) / machines_per_work_center)
      world.work_centers = [Work_Center() for _ in range(work_center_count)]
      world.machines = []

      for work_center_index, work_center in enumerate(world.work_centers, 1):
        work_center.id = work_center_index - 1
        work_center.name = 'Work Center %d' % work_center_index
        work_center.machines = []
        for offset in range(machines_per_work_center):
          machine_number = (work_center_index - 1) * machines_per_work_center + offset + 1
          if machine_number > len(machine_ids):
            break
          machine = Machine()
          machine.id = machine_number - 1
          machine.name = 'Machine %d,%d' % (work_center_index, offset + 1)
          machine.scale = int(float(data[f'scale parameter´s value of machine {machine_number}']))
          machine.shape = float(data[f'shape parameter´s value of machine {machine_number}'])
          machine.failure_cost = int(float(data[f'failure cost of machine {machine_number}']))
          machine.CM_duration = int(float(data[f'corrective maintenance duration of machine {machine_number}']))
          machine.PM_cost = int(float(data[f'pm cost of machine {machine_number}']))
          machine.PM_duration = int(float(data[f'pm duration of machine {machine_number}']))
          machine.energy_consumption = float(data[f'energy consumption of machine {machine_number}'])
          machine.idle_energy_consumption = float(data[f'idle energy consumption of machine {machine_number}'])
          machine.threshold = machine.initial_threshold = int(weibull_min.ppf(0.9, machine.shape, loc=0, scale=machine.scale))
          machine.operations_to_process = []
          work_center.machines.append(machine)
          world.machines.append(machine)

      # Gather the opaque machine-address tokens written by the original dataset.
      machine_tokens = []
      parsed_available_machines = {}
      job_count = int(float(data['number of jobs']))
      for job_index in range(1, job_count + 1):
        operation_count = int(float(data[f'operations of job {job_index}']))
        for operation_index in range(1, operation_count + 1):
          key = f'operation {operation_index} of job {job_index} available machines'
          entries = self._parse_available_machine_entries(data[key])
          parsed_available_machines[key] = entries
          for machine_token, _ in entries:
            if machine_token not in machine_tokens:
              machine_tokens.append(machine_token)

      uses_stable_machine_ids = all(token.startswith("machine:") for token in machine_tokens)
      if uses_stable_machine_ids:
        machine_by_token = {f"machine:{machine.id + 1}": machine for machine in world.machines}
        unknown_tokens = [token for token in machine_tokens if token not in machine_by_token]
        if unknown_tokens:
          raise ValueError(f"Dataset references unknown machine IDs: {unknown_tokens}")
      else:
        if len(machine_tokens) != len(world.machines):
          raise ValueError(f"Dataset has {len(machine_tokens)} machine tokens but {len(world.machines)} machine parameter blocks.")

        if machine_address_strategy == "sorted":
          ordered_tokens = sorted(machine_tokens, key=lambda value: int(value, 16))
        elif machine_address_strategy == "first_seen":
          ordered_tokens = machine_tokens
        else:
          raise ValueError("machine_address_strategy must be 'first_seen' or 'sorted'.")
        machine_by_token = {token: machine for token, machine in zip(ordered_tokens, world.machines)}

      world.jobs = [Job() for _ in range(job_count)]
      operation_id = 0
      for job_index, job in enumerate(world.jobs, 1):
        job.name = 'Job %d' % job_index
        job.id = job_index - 1
        job.tardiness_cost = float(data[f'unit tardiness cost of job {job_index}'])
        job.arrival_time = int(float(data[f'arrival time of job {job_index}']))
        operation_count = int(float(data[f'operations of job {job_index}']))
        job.operations = [Operation() for _ in range(operation_count)]

        for operation_index, operation in enumerate(job.operations, 1):
          operation.id = operation_id
          operation_id += 1
          operation.name = 'Operation %d,%d' % (job_index, operation_index)
          operation.parent = job
          operation.order = operation_index - 1
          key = f'operation {operation_index} of job {job_index} available machines'
          available_machine_tokens = parsed_available_machines[key]
          operation.available_machines = {machine_by_token[address]: processing_time for address, processing_time in available_machine_tokens}
          operation.average_processing_time = np.mean(list(operation.available_machines.values()))
          for machine, processing_time in operation.available_machines.items():
            machine.operations_to_process.extend([(operation, processing_time)])

        job.processing_time = sum(int(operation.average_processing_time) for operation in job.operations)
        job.remaining_processing_time = job.processing_time
        job.due_date = int(job.arrival_time + job.processing_time * world.DDT)

      world.operations = sum(len(job.operations) for job in world.jobs)
      self.reset_world(world)
      print("Loaded dataset:", dataset_path, "DDT:", world.DDT, "lambda:", world._lambda,
            "jobs:", len(world.jobs), "machines:", len(world.machines), "operations:", world.operations)
      return world

  def reset_world(self, world):
      # set initial states
      self.n_actions_agents = [len(agent.operations_to_process) for agent in world.machines] # Number of actions per machine
      world.total_operations = sum([len(job.operations) for job in world.jobs]) # Total number of operations
      world.operations_done = []  # Reset the list of completed operations
      world.new_job = [] # Reset the list of new jobs
      world.ready_machine = world.machines[0] # Set the first machine as ready (arbitrary choice)
      # Set a ready operation from jobs that have arrived
      world.ready_operation = next((job.operations[0] for job in world.jobs if job.arrival_time == 0), None)
      # List of attributes to set to 0
      attributes_to_reset = ['total_cost', 'total_timestamp', 'total_tardiness_cost', 'tardiness_rate', 'total_energy_consumption', 'total_maintenance_cost', 'previous_tardiness_rate',
                             'MRT_DR_O', 'LRT_DR_O', 'SPT_DR_O', 'EDD_DR_O', 'MCR_DR_O', 'SLK_DR_O', 'CR_DR_O', 'ATC_DR_O', 'LSPO_DR_O']
      # Loop through the list and set each attribute to 0
      for attr in attributes_to_reset:
        setattr(world, attr, 0)

      # Reset job-level attributes
      for job in world.jobs:
        job.CRJ = job.op_assigned = job.op_processed = 0 # Reset progress counters
        job.cancelled= False  # Reset cancelled flag
        job.remaining_processing_time = job.processing_time  # Reset remaining processing time
        job.tardy_operations = job.finish_processing = 0  # Reset tardiness and finish time
        job.inserted = job.legal = job.arrival_time == 0  # Set 'inserted' and 'legal' based on arrival time

       # List of machine-level attributes to reset
      attributes_to_reset = ['total_energy_consumption', 'workload', 'virtual_age', 'assigned_op', 'latest_PM', 'buffer_time',
                             'total_maintenance_cost', 'waiting_time', 'utilization', 'assigned_op', 'PM_actions_counts', 'CM_actions_counts']

      for machine_id, machine in enumerate(world.machines):
        # Number of actions (operations) available to the machine at the current time
        machine.actions =  sum(1 for op in machine.operations_to_process if op[0].parent.arrival_time <= world.total_timestamp)
        machine.buffer = {} # Reset the machine's buffer
        # Set legal actions for the machine (operations that are ready to be processed)
        machine.legal_actions = [operation for operation in machine.operations_to_process if operation[0].order == 0 and operation[0].parent.legal]
        machine.request = any(machine.legal_actions) # Set request flag based on availability of legal actions
        machine.availability = 1 # Set machine availability to 'available'
        machine.threshold = machine.initial_threshold # Reset failure threshold
        machine.action = [0, 0, 0]  # Reset current action

        # Reset each machine attribute to 0
        for attr in attributes_to_reset:
          setattr(machine, attr, 0)

  # Method to calculate the expected tardiness rate
  def expected_tardiness_rate(self, world, filtered_jobs):
    expected_tardy_operations = 0 # Initialize counter for expected tardy operations
    Total = sum([len(job.operations) for job in filtered_jobs]) # Total number of operations in the filtered jobs
    average_CT = int(np.mean([machine.action[1] for machine in world.machines])) # Average completion time of machines
    # Iterate over each job to estimate tardy operations
    for job in filtered_jobs:
      # Check if the job is expected to be tardy based on its due date and remaining processing time
      if world.total_timestamp < job.due_date <= world.total_timestamp + average_CT + job.remaining_processing_time:
        total_time = 0 # Accumulator for operation processing times
        expected_tardiness_time = world.total_timestamp + average_CT + job.remaining_processing_time - job.due_date
        # Iterate over operations in reverse order to calculate expected tardy operations
        for op in reversed(job.operations):
          total_time += int(op.average_processing_time)
          if total_time <= expected_tardiness_time:
            expected_tardy_operations += 1  # Increment the counter
    return expected_tardy_operations/Total # Return the expected tardiness rate

  def actual_tardiness_rate(self, world, jobs):
    Ntard = 0 # Counter for tardy operations
    Total = sum([len(job.operations) for job in jobs]) # Total number of operations
    Ntard = sum([job.tardy_operations for job in jobs]) # Sum of tardy operations across jobs
    # Add operations from jobs that haven't started and are past their due date
    Ntard += sum([len(job.operations) for job in jobs if job.remaining_processing_time == job.processing_time and world.total_timestamp >= job.due_date])
    return Ntard/Total # Return the actual tardiness rate

  # Reward function based on the change in tardiness rates
  def reward(self, world):
    #Total tardiness rates at the current decision time
    if all(job.CRJ==1 for job in world.jobs):
      jobs_in_the_buffer = [job for job in world.jobs if job.arrival_time <= world.total_timestamp]
    else:
      # Jobs that are available and have operations ready to be scheduled
      jobs_in_the_buffer = [op[0].parent for op in world.ready_machine.legal_actions if op[0].parent.arrival_time <= world.total_timestamp]

    # Calculate actual and expected tardiness rates
    actual_tardiness_rate = self.actual_tardiness_rate(world, jobs_in_the_buffer)
    expected_tardiness_rate = self.expected_tardiness_rate(world, jobs_in_the_buffer)
    world.tardiness_rate = actual_tardiness_rate + expected_tardiness_rate # Update the world's tardiness rate

    if maintenance_integrated:
      #Total PM costs until the current decision time
      world.total_maintenance_cost = sum(machine.total_maintenance_cost for machine in world.machines)

    #Total energy consumption until the current decision time
    world.total_energy_consumption = sum(machine.total_energy_consumption for machine in world.machines)

    #Total tardiness cost until the current decision time
    world.total_tardiness_cost = sum(job.tardiness_cost * job.tardy_operations for job in world.jobs)

    world.total_cost = world.total_tardiness_cost + world.total_energy_consumption + world.total_maintenance_cost

    # Sharp reward (Minimizing the Tardiness rate)
    if actual_tardiness_rate + expected_tardiness_rate > world.previous_tardiness_rate:
      return -0.75  # Negative reward if tardiness rate increased
    elif actual_tardiness_rate + expected_tardiness_rate < world.previous_tardiness_rate:
      return 1  # Positive reward if tardiness rate decreased
    else:
      return -0.5 # Small negative reward if tardiness rate remained the same

  
  # Method to construct the observation vector for the agent
  def observation(self, world):
      if maintenance_integrated:
        # How many times PM was performed on the shop floor
        PM_actions_counts = [machine.PM_actions_counts for machine in world.machines]
        # How many times CM was performed on the shop floor
        CM_actions_counts = [machine.CM_actions_counts for machine in world.machines]
        # Mean Failure risk of machines
        failure_risks = [failure_risk(machine) for machine in world.machines]
        # How many times PM was performed on the current machine
        PM_actions_counts_machine = world.ready_machine.PM_actions_counts
        # How many times CM was performed on the current machine
        CM_actions_counts_machine = world.ready_machine.CM_actions_counts
        # Failure risk of the current machine
        failure_risk_machine = failure_risk(world.ready_machine)
      #Average of machine utilization rate from the perspective of the completed operations
      machine_utilization_op = [machine.assigned_op/machine.actions for machine in world.machines]
      mean_utilization_op_rate = np.mean(machine_utilization_op)
      #std of machine utilization rate from the perspective of the completed operations
      std_utilization_op_rate = np.std(machine_utilization_op)
      #Average of energy consumption
      energy_consumption = [machine.total_energy_consumption for machine in world.machines]
      #Normalized energy consumption
      EC_machine = (world.ready_machine.total_energy_consumption - min(energy_consumption)) / (max(energy_consumption) - min(energy_consumption)) if max(energy_consumption) != 0 else 0
      average_energy_consumption = np.mean(energy_consumption) / (10 * world.total_timestamp) if world.total_timestamp != 0 else 0
      #Info related to jobs
      jobs_in_the_buffer = [op[0].parent for op in world.ready_machine.legal_actions if op[0].parent.arrival_time <= world.total_timestamp]
      #Completion rate of all the operations
      total_operations = sum(len(job.operations) for job in jobs_in_the_buffer)
      completion_rate = sum([job.op_processed for job in jobs_in_the_buffer]) / total_operations
      CRJ_op = [job.CRJ for job in jobs_in_the_buffer]
      #Average completion rate from the perspective of the number of completed operations
      average_CRJ_op = np.mean(CRJ_op)
      #Std completion rate from the perspective of the number of completed operations
      std_CRJ_op = np.std(CRJ_op)
      #Average completion rate from the perspective of the completed processed time
      CRJ_time = [(job.processing_time - job.remaining_processing_time) / (job.processing_time) for job in jobs_in_the_buffer]
      average_CRJ_time = np.mean(CRJ_time)
      #Std completion rate from the perspective of the completed processed time
      std_CRJ_time = np.std(CRJ_time)
      #Total operations tardiness rate at the current decision time
      actual_tardiness_rate = self.actual_tardiness_rate(world, jobs_in_the_buffer)
      expected_tardiness_rate = self.expected_tardiness_rate(world, jobs_in_the_buffer)
      world.previous_tardiness_rate = actual_tardiness_rate + expected_tardiness_rate
      #Actual Job tardiness rate
      actual_tardiness_rate_job = sum([1 for job in jobs_in_the_buffer if job.tardy_operations > 0]) / len(jobs_in_the_buffer)
      #Expected Job tardiness rate
      average_CT = int(np.mean([machine.action[1] for machine in world.machines]))
      expected_tardiness_rate_job = sum([1 for job in jobs_in_the_buffer if world.total_timestamp < job.due_date <= world.total_timestamp + average_CT + job.remaining_processing_time]) / len(jobs_in_the_buffer)
      # Maximal/Minimal remaining time of the candidate jobs
      remaining_time = [job.remaining_processing_time/job.processing_time for job in jobs_in_the_buffer]
      # Minimal critial ratio
      min_critical_ratio = min([(job.due_date - world.total_timestamp) / job.remaining_processing_time if job.remaining_processing_time != 0 else 0 for job in jobs_in_the_buffer])
      if maintenance_integrated:
        return np.array([PM_actions_counts_machine, CM_actions_counts_machine, failure_risk_machine, average_CT, average_energy_consumption, EC_machine, completion_rate, actual_tardiness_rate, expected_tardiness_rate, mean_utilization_op_rate, std_utilization_op_rate, average_CRJ_op, std_CRJ_op, average_CRJ_time, std_CRJ_time], dtype=np.float32)
      else:
        return np.array([1/world._lambda, world.DDT, len(world.jobs), average_CT, completion_rate, actual_tardiness_rate_job, expected_tardiness_rate_job, actual_tardiness_rate, expected_tardiness_rate, average_CRJ_op, std_CRJ_op, average_CRJ_time, std_CRJ_time, min_critical_ratio], dtype=np.float32)


# ---- Extracted notebook cell 28 ----
class Memory(object):
    def __init__(self, capacity: int, alpha=0.6):
        # Initialize the capacity of the replay buffer
        self.capacity = capacity
        # Initialize the buffer using deque with a maximum length of capacity
        self.buffer = deque(maxlen=self.capacity)
        # Initialize a deque to store the priorities of experiences
        self.priorities = deque(maxlen=self.capacity)
        # Keep track of the number of entries currently in the buffer
        self.n_entries = 0
        # Set the alpha parameter for controlling prioritization degree (0 - no prioritization, 1 - full prioritization)
        self.alpha = alpha
        
    # Method to add a new experience to the buffer
    def add(self, transitions):
        # Find the maximum priority in the current buffer; default to 1.0 if the buffer is empty
        max_priority = max(self.priorities) if self.buffer else 1.0
        # Append the new transition to the buffer
        self.buffer.append(transitions)
        # Assign the maximum priority to the new transition
        self.priorities.append(max_priority)
        # Increment the number of entries if capacity is not yet reached
        if self.n_entries < self.capacity:
            self.n_entries += 1
    # Method to sample a batch of experiences from the buffer
    def sample(self, batch_size, beta=0.4):
        # If the buffer is full, use all priorities; else, use priorities up to n_entries
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities)[:self.n_entries]

        # Calculate the sampling probabilities by raising priorities to the power of alpha
        probabilities = priorities ** self.alpha
        # Normalize the probabilities so they sum to 1
        probabilities /= probabilities.sum()

        # Randomly sample indices based on the probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        # Retrieve the samples corresponding to the sampled indices
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance-sampling weights to correct for bias introduced by prioritized sampling
        total = len(self.buffer)
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        # Normalize weights so that the maximum weight is 1
        weights /= weights.max()
        # Convert weights to a NumPy array of type float32
        weights = np.array(weights, dtype=np.float32)

        # Unzip the samples to separate components (e.g., states, actions, rewards, next_states, dones)
        batch = list(zip(*samples))
        # Return the batch of samples, their indices, and the importance-sampling weights
        return batch, indices, weights

    # Method to update the priorities of sampled experiences after learning
    def update_priorities(self, batch_indices, batch_priorities):
        # Iterate over the indices and corresponding new priorities
        for idx, priority in zip(batch_indices, batch_priorities):
            # Update the priority of the experience at index idx
            self.priorities[idx] = priority

    # Method to get the current number of samples in the buffer
    def __len__(self):
        # Return the length of the buffer
        return len(self.buffer)


# ---- Extracted notebook cell 30 ----
class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, neurons_per_layer):
        # Initialize the superclass (nn.Module)
        super(DuelingNetwork, self).__init__()
        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize an empty list to store the layers
        layers = []
        # Set the initial input size to the state size
        input_size = state_size
        # Loop over the number of hidden layers to build the network dynamically
        for i in range(hidden_layers):
            # Add a linear (fully connected) layer with specified input and output sizes
            layers.append(nn.Linear(input_size, neurons_per_layer[i]))
            # Add a Layer Normalization layer to normalize the inputs across the features
            layers.append(nn.LayerNorm(neurons_per_layer[i]))
            # Add a LeakyReLU activation function to introduce non-linearity
            layers.append(nn.LeakyReLU())
            # Add a Dropout layer for regularization to prevent overfitting (20% dropout rate)
            layers.append(nn.Dropout(p=0.2))
            # Update the input size for the next layer to be the size of the current layer
            input_size = neurons_per_layer[i]

        # Update the input size for the next layer to be the size of the current layer
        self.feature_layer = nn.Sequential(*layers).to(self.device)
        # Define the value stream, which outputs a single scalar value
        self.value_stream = nn.Linear(input_size, 1).to(self.device)
        # Define the advantage stream, which outputs a vector of size equal to the number of actions
        self.advantage_stream = nn.Linear(input_size, action_size).to(self.device)

    # Define the forward pass of the network
    def forward(self, x):
        # Ensure the input x is a PyTorch tensor of type float32 and move it to the device
        x = torch.as_tensor(x, dtype=torch.float32).to(self.device)
        # Pass the input through the feature extraction layers
        x = self.feature_layer(x)
        # Compute the value from the value stream
        value = self.value_stream(x)
        # Compute the advantages from the advantage stream
        advantage = self.advantage_stream(x)
        # Combine value and advantages to get the Q-values using the dueling architecture formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        # Return the computed Q-values
        return q_values


# ---- Extracted notebook cell 32 ----
class DQNAgent:
    def __init__(self, input_dim, action_size, hidden_layers, neurons_per_layer, batch_size):
        # Initialize the agent with state dimension, action size, network architecture, and batch size
        self.state_size = input_dim      # Dimension of the input state vector
        self.action_size = action_size   # Number of possible actions 
        self.batch_size = batch_size     # Number of samples per training batch 
        super(DQNAgent, self).__init__()

        # Parameters for training
        self.train_start = 1000          # Number of experiences to collect before starting training     
        self.memory_size = 100000        # Maximum capacity of the replay memory
        self.memory = Memory(self.memory_size)  # Initialize the replay memory
        self.gamma = 0.99               # Discount factor for future rewards
        self.epsilon = 1.0              # Initial exploration rate for epsilon-greedy policy
        self.epsilon_min = 0.01          # Minimum exploration rate
        self.loss_fn = nn.MSELoss(reduction='none')   # Mean Squared Error loss function without reduction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Use GPU if available

        # Initialize the main network (Dueling DQN)
        self.model = DuelingNetwork(input_dim, action_size, hidden_layers, neurons_per_layer).to(self.device)
        self.model.apply(self.weights_init)                        # Apply custom weight initialization

        # Initialize the target network with the same weights as the main network
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Exploration parameters
        self.explore_step = 80000    # Number of steps over which to anneal epsilon (not used in this code)
        self.epsilon_decay = 0.995   # Decay rate for epsilon after each episode 
        self.tau = 0.01              # Soft update parameter for target network
        self.update_rate = 1000      # Number of steps between target network updates
        self.learn_step_counter = 0  # Counter for learning steps

        # Optimizer and learning rate scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def weights_init(self, m):
        # Custom weight initialization function
        classname = m.__class__.__name__  # Get the name of the class
        if classname.find('Linear') != -1:
            # If the layer is a Linear layer, apply Xavier uniform initialization
            torch.nn.init.xavier_uniform_(m.weight)

    def soft_update(self):
        # Soft update the target network parameters towards the main network parameters
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            # Update target parameter: θ_target = τ * θ_local + (1 - τ) * θ_target
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update(self):
        # Update the network parameters using a batch of experiences from the replay memory
        if self.learn_step_counter % self.update_rate == 0:
            # Every 'update_rate' steps, perform a soft update of the target network
            self.soft_update()
        self.learn_step_counter += 1  # Increment the learning step counter

        # Sample a batch of experiences with prioritization
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # Convert the batch data to PyTorch tensors and move to the appropriate device (CPU/GPU)
        states = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions, dtype=np.int64), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones, dtype=np.float32), dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Compute the current Q-values from the main network
        q_values = self.model(states)
        # Select the Q-values corresponding to the taken actions
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the next actions using the main network (for Double DQN)
        next_actions = self.model(next_states).argmax(1)
        # Compute the next Q-values from the target network using the next actions
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Compute the expected Q-values using the Bellman equation
        expected_q_values = rewards + self.gamma * (1 - dones) * next_q_values.detach()

        # Compute the Temporal Difference (TD) errors
        td_errors = expected_q_values - q_values
        # Compute the loss, weighted by importance-sampling weights
        loss = (weights * self.loss_fn(q_values, expected_q_values)).mean()

        # Perform a gradient descent step
        self.optimizer.zero_grad()    # Zero the gradients
        loss.backward()               # Backpropagate the loss
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step() # Update the network parameters

        # Update the priorities in the replay memory based on the TD errors
        priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.memory.update_priorities(indices, priorities)

        # Decay the exploration rate epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def append_sample(self, state, action, reward, next_state, done):
        # Add a new experience to the replay memory
        self.memory.add((state, action, reward, next_state, done))

    def get_action(self, state, world):
         # Select an action using an epsilon-greedy policy
        self.model.eval()   # Set the model to evaluation mode (disables dropout, etc.)
        with torch.no_grad():
            if np.random.rand() <= self.epsilon:
                # With probability epsilon, select a random action (exploration)
                action = random.randrange(self.action_size)
            else:
                # Otherwise, select the action with the highest predicted Q-value (exploitation)
                state = torch.tensor(np.array([state], dtype=np.float32), dtype=torch.float32).to(self.device)
                q_values = self.model(state)  # Get Q-values from the model
                action = torch.argmax(q_values, dim=-1).item()   # Select the action with the highest Q-value
        self.model.train()  # Set back to training mode
        return action

    def save_model(self, filename):
        # Save the model parameters to a file
        torch.save(self.model.state_dict(), filename)

    # def load_model(self, filename):
    #     # Load the model parameters from a file
    #     self.model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    #     self.model.eval() # Set the model to evaluation mode after loading
    def load_model(self, filename):
        state_dict = torch.load(filename, map_location=torch.device('cpu'), weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()


# ---- Extracted notebook cell 34 ----
def train_agents(hidden_layers, neurons_per_layer, batch_size, episodes=1000, eval_every=25, dataset_path=None):

    scores = []
    best_score = -np.inf  # Initialize best_score appropriately
    EPISODES = episodes

    agent = DQNAgent(input_dim=state_dim, action_size=output_dim, hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer, batch_size=batch_size)

    print("........................................Collecting Experience........................................")

    for e in range(EPISODES):
        env = make_env(dataset_path=dataset_path)
        world = env.world
        done = False
        reward_epo = 0  # Cumulative reward of the episode

        while not done:
            # Check for available jobs and idle machines
            if not (any(job.legal for job in world.jobs) and any(machine.request for machine in world.machines)):
                env.update_state()  # Move the shop floor forward if no machines are ready or no legal jobs
                continue

            world.ready_machine = next((machine for machine in random.sample(world.machines, len(world.machines)) if machine.request), None)
            env.get_legal_actions(world.ready_machine)  # Update the legal operations for the machine that requests processing

            if world.ready_machine.legal_actions:
                state = env._get_obs
                action = agent.get_action(state, world)
                next_state, reward, done, _, _ = env.step(action)
                agent.append_sample(state, action, reward, next_state, done)
                reward_epo += reward  # Accumulate the reward
                # Update the agent if enough experiences have been collected
                if agent.memory.n_entries >= agent.train_start:
                    agent.update()
            else:
                world.ready_machine.request = False  # No candidate operation, wait for the next decision time

        scores.append(reward_epo)  # Append the cumulative episode reward
        mean_utilization = np.mean([machine.utilization for machine in world.machines]) * 100

        print("episode: {}/{}, episode reward: {:.2f}, tardiness rate: {:.2f}, mean Machine Utilization: {:.2f}%".format(
            e, EPISODES, reward_epo, world.tardiness_rate, mean_utilization))

        # Save the model if the cumulative episode reward is better
        if reward_epo > best_score:
            best_score = reward_epo
            agent.save_model('Best_agent_hidden_layers_{}neurons_per_layer_{}_batch_size_{}.pth'.format(hidden_layers, neurons_per_layer, batch_size))

        scheduling_scheme = env.render()

        fig1 = Selected_heuristics_operations(world)

        # Display the utilization of each machine in a histogram
        fig2 = px.bar(x=[machine.name for machine in world.machines], y=[machine.utilization for machine in world.machines])

        if maintenance_integrated:
            total_PM_actions_counts = np.sum([machine.PM_actions_counts for machine in world.machines])
            total_CM_actions_counts = np.sum([machine.CM_actions_counts for machine in world.machines])
            wandb.log({'Number of jobs': len(world.jobs), 'Number of CM actions ': total_CM_actions_counts, 'Number of PM actions ': total_PM_actions_counts, 'Total cost of energy consumption': world.total_energy_consumption, 'Total maintenance cost': world.total_maintenance_cost,
                       'Reward': reward_epo, 'Total cost': world.total_cost, 'Tardiness rate': world.tardiness_rate, 'Tardiness cost': world.total_tardiness_cost, 'Number of selected DR': fig1, 'Utilization rate by machine': fig2, 'Exploration': agent.epsilon, "Scheduling scheme": scheduling_scheme})
        else:
            wandb.log({'Number of jobs': len(world.jobs), 'Reward': reward_epo, 'Tardiness rate': world.tardiness_rate, 'Tardiness cost': world.total_tardiness_cost, 'Total cost of energy consumption': world.total_energy_consumption, 'Number of selected DR': fig1, 'Utilization rate by machine': fig2, 'Exploration': agent.epsilon, "Scheduling scheme": scheduling_scheme})

        env.close()
    return best_score


# ---- Extracted notebook cell 36 ----
def Selected_heuristics_operations(world): # Selected heuristics for operation sequencing
  DR_actions = ['MRT_DR_O', 'LRT_DR_O', 'SPT_DR_O', 'EDD_DR_O', 'MCR_DR_O', 'CR_DR_O', 'ATC_DR_O', 'LSPO_DR_O', 'SLK_DR_O']
  traces = [go.Bar(x=DR_actions, y=[getattr(world, action) for action in DR_actions], name="Operation Sequencing")]
  layout = go.Layout(xaxis=dict(title='operation action'), yaxis=dict(title='Counts'))
  fig = go.Figure(data=traces, layout=layout)
  return fig


# ---- Extracted notebook cell 38 ----
# Define the objective function for Optuna
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_layers = trial.suggest_int("hidden_layers", 3, 7)
    neurons_per_layer = [trial.suggest_int(f"neurons_per_layer_{i}", 64, 256) for i in range(hidden_layers)]
    episodes = trial.suggest_int("episodes", 100, 1000, step=100)

    mean_reward = train_agents(hidden_layers, neurons_per_layer, batch_size, episodes=episodes)

    return mean_reward
