from pdb import set_trace as T

import os

import nmmo
from nmmo.core import config

import rllib_wrapper
import tasks

from scripted import baselines

class RLlibConfig(config.Achievement):
   '''Base config for RLlib Models

   Extends core Config, which contains environment, evaluation,
   and non-RLlib-specific learning parameters

   IMPORTANT: Configure NUM_GPUS and NUM_WORKERS for your hardware
   Note that EVALUATION_NUM_WORKERS cores are reserved for evaluation
   and one additional core is reserved for the driver process.
   Therefore set NUM_WORKERS <= cores - EVALUATION_NUM_WORKERS - 1
   '''

   #Run in train/evaluation mode
   EVALUATE     = False
   N_TRAIN_MAPS = 256

   @property
   def MODEL(self):
      return self.__class__.__name__

   @property
   def PATH_MAPS(self):
      maps = super().PATH_MAPS
      if self.EVALUATE:
          self.TERRAIN_FLIP_SEED = True
          return os.path.join(maps, 'evaluation')
      return os.path.join(maps, 'training')

   @property
   def NMAPS(self):
      if not self.EVALUATE:
          return self.N_TRAIN_MAPS
      return super().NMAPS

   #Checkpointing. Resume will load the latest trial, e.g. to continue training
   #Restore (overrides resume) will force load a specific checkpoint (e.g. for rendering)
   EXPERIMENT_DIR          = 'experiments'
   RESUME                  = False

   RESTORE                 = True
   RESTORE_ID              = 'Baseline' #Experiment name suffix
   RESTORE_CHECKPOINT      = 1000

   #Policy specification
   EVAL_AGENTS             = [baselines.Meander, baselines.Forage, baselines.Combat, nmmo.Agent]
   AGENTS                  = [nmmo.Agent]
   TASKS                   = []

   #Hardware and debug
   NUM_GPUS_PER_WORKER     = 0
   NUM_GPUS                = 0
   EVALUATION_NUM_WORKERS  = 3
   LOCAL_MODE              = False
   LOG_LEVEL               = 1

   #Training and evaluation settings
   EVALUATION_INTERVAL     = 1
   EVALUATION_NUM_EPISODES = 3
   EVALUATION_PARALLEL     = True
   TRAINING_ITERATIONS     = 1000
   KEEP_CHECKPOINTS_NUM    = 3
   CHECKPOINT_FREQ         = 1
   LSTM_BPTT_HORIZON       = 16
   NUM_SGD_ITER            = 1

   #Model
   SCRIPTED                = None
   N_AGENT_OBS             = 100
   NPOLICIES               = 1
   HIDDEN                  = 64
   EMBED                   = 64

   #Reward
   COOPERATIVE             = False
   TEAM_SPIRIT             = 0.0

class Small(RLlibConfig, config.Small):
   '''Small scale Neural MMO training setting

   Features up to 64 concurrent agents and 32 concurrent NPCs,
   64 x 64 maps (excluding the border), and 128 timestep horizons'''
   
   
   #Memory/Batch Scale
   NUM_WORKERS             = 4
   TRAIN_BATCH_SIZE        = 64 * 256 * NUM_WORKERS
   ROLLOUT_FRAGMENT_LENGTH = 128
   SGD_MINIBATCH_SIZE      = 128
 
   #Horizon
   TRAIN_HORIZON           = 128
   EVALUATION_HORIZON      = 128


class Medium(RLlibConfig, config.Medium):
   '''Medium scale Neural MMO training setting

   Features up to 256 concurrent agents and 128 concurrent NPCs,
   128 x 128 maps (excluding the border), and 1024 timestep horizons'''
 
   #Memory/Batch Scale
   NUM_WORKERS             = 4
   TRAIN_BATCH_SIZE        = 64 * 256 * NUM_WORKERS
   ROLLOUT_FRAGMENT_LENGTH = 256
   SGD_MINIBATCH_SIZE      = 128
 
   #Horizon
   TRAIN_HORIZON           = 1024
   EVALUATION_HORIZON      = 1024


class Large(RLlibConfig, config.Large):
   '''Large scale Neural MMO training setting

   Features up to 2048 concurrent agents and 1024 concurrent NPCs,
   1024 x 1024 maps (excluding the border), and 8192 timestep horizons'''
 
   #Memory/Batch Scale
   NUM_WORKERS             = 2
   TRAIN_BATCH_SIZE        = 64 * 256 * NUM_WORKERS
   ROLLOUT_FRAGMENT_LENGTH = 32
   SGD_MINIBATCH_SIZE      = 128

   #Horizon
   TRAIN_HORIZON           = 8192
   EVALUATION_HORIZON      = 8192



class Debug(Small, config.AllGameSystems):
   '''Debug Neural MMO training setting

   A version of the SmallMap setting with greatly reduced batch parameters.
   Only intended as a tool for identifying bugs in the model or environment'''

   TASKS                   = tasks.All

   RESTORE                 = False
   NUM_WORKERS             = 1

   TRAINING_ITERATIONS     = 2

   SGD_MINIBATCH_SIZE      = 100
   TRAIN_BATCH_SIZE        = 400
   TRAIN_HORIZON           = 200
   EVALUATION_HORIZON      = 50

   HIDDEN                  = 2
   EMBED                   = 2


### AICrowd competition settings
class CompetitionRound1(Medium, config.AllGameSystems):

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   TASKS                   = tasks.All
   NENT                    = 128
   NPOP                    = 1

class CompetitionRound2(Medium, config.AllGameSystems):

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   @property
   def NENT(self):
      return 8 * len(self.AGENTS)

   NPOP                    = 16
   EVAL_AGENTS             = 8*[baselines.Meander, baselines.Forage, baselines.Combat, nmmo.Agent]
   AGENTS                  = NPOP*[nmmo.Agent]
   TASKS                   = tasks.All

   AGENT_LOADER            = config.TeamLoader
   COOPERATIVE             = True
   TEAM_SPIRIT             = 1.0

class CompetitionRound3(Large, config.AllGameSystems):

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   NENT                    = 1024
   NPOP                    = 32
   COOPERATIVE             = True
   TEAM_SPIRIT             = 1.0
   AGENT_LOADER            = config.TeamLoader
   TASKS                   = tasks.All


### NeurIPS Experiments
class SmallAllSystems(Small, config.AllGameSystems):
   TASKS                   = tasks.All

class MediumAllSystems(Medium, config.AllGameSystems):
   TASKS                   = tasks.All

class LargeAllSystems(Large, config.AllGameSystems):
   TASKS                   = tasks.All

class DomainRandomization(Medium, config.AllGameSystems): pass
class DomainRandomization16384(DomainRandomization):
   N_TRAIN_MAPS            = 16384
class DomainRandomization256(DomainRandomization):
   N_TRAIN_MAPS            = 256
class DomainRandomization32(DomainRandomization):
   N_TRAIN_MAPS            = 32
class DomainRandomization1(DomainRandomization):
   N_TRAIN_MAPS            = 1

class MagnifyExploration(Medium, config.Resource, config.Progression): pass
class Population4(MagnifyExploration):
   NENT                    = 4
class Population32(MagnifyExploration):
   NENT                    = 32
class Population256(MagnifyExploration):
   NENT                    = 256

class TeamBased(MagnifyExploration, config.Combat):
   NENT                    = 128
   NPOP                    = 32
   COOPERATIVE             = True
   TEAM_SPIRIT             = 0.5

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT
