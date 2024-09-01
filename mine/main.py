import pgx
from functools import partial
from core.testing.utils import render_pgx_2p
from core.memory.replay_memory import EpisodeReplayBuffer
import jax.numpy as jnp
from core.evaluators.evaluation_fns import make_nn_eval_fn_no_params_callable
from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS

from functools import partial
from core.networks.azresnet import AZResnetConfig, AZResnet
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
import optax
from core.types import StepMetadata

env = pgx.make("othello")

def state_to_nn_input(state):
    return state.observation


def step_fn(state, action):
    new_state = env.step(state, action)
    return new_state, StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count,
    )


def init_fn(key):
    state = env.init(key)
    return state, StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
    )


resnet = AZResnet(
    AZResnetConfig(
        policy_head_out_size=env.num_actions,
        num_blocks=4,
        num_channels=32,
    )
)

# alphazero can take an arbirary search `backend`
# here we use classic MCTS
az_evaluator = AlphaZero(MCTS)(
    eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
    num_iterations=32,
    max_nodes=40,
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(),
    temperature=1.0,
)

az_evaluator_test = AlphaZero(MCTS)(
    eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
    num_iterations=64,
    max_nodes=80,
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(),
    temperature=0.0,
)


model = pgx.make_baseline_model("othello_v0")

baseline_eval_fn = make_nn_eval_fn_no_params_callable(model, state_to_nn_input)

baseline_az = AlphaZero(MCTS)(
    eval_fn=baseline_eval_fn,
    num_iterations=64,
    max_nodes=80,
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(),
    temperature=0.0,
)


def greedy_eval(obs):
    value = (obs[..., 0].sum() - obs[..., 1].sum()) / 64
    return jnp.ones((1, env.num_actions)), jnp.array([value])


greedy_baseline_eval_fn = make_nn_eval_fn_no_params_callable(greedy_eval, state_to_nn_input)


greedy_az = AlphaZero(MCTS)(
    eval_fn=greedy_baseline_eval_fn,
    num_iterations=64,
    max_nodes=80,
    branching_factor=env.num_actions,
    action_selector=PUCTSelector(),
    temperature=0.0,
)


replay_memory = EpisodeReplayBuffer(capacity=1000)


render_fn = partial(render_pgx_2p, p1_label="Black", p2_label="White", duration=900)



trainer = Trainer(
    batch_size=128,
    train_batch_size=128,
    warmup_steps=0,
    collection_steps_per_epoch=256,
    train_steps_per_epoch=64,
    nn=resnet,
    loss_fn=partial(az_default_loss_fn, l2_reg_lambda=0.0),
    optimizer=optax.adam(1e-3),
    evaluator=az_evaluator,
    memory_buffer=replay_memory,
    max_episode_steps=80,
    env_step_fn=step_fn,
    env_init_fn=init_fn,
    state_to_nn_input_fn=state_to_nn_input,
    testers=[
        TwoPlayerBaseline(
            num_episodes=128,
            baseline_evaluator=baseline_az,
            render_fn=render_fn,
            render_dir=".",
            name="pretrained",
        ),
        TwoPlayerBaseline(
            num_episodes=128,
            baseline_evaluator=greedy_az,
            render_fn=render_fn,
            render_dir=".",
            name="greedy",
        ),
    ],
    evaluator_test=az_evaluator_test,
    # wandb_project_name = 'turbozero-othello'
)

output = trainer.train_loop(seed=0, num_epochs=100, eval_every=5)
