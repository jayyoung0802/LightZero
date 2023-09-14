from easydict import EasyDict

env_name = 'smac'
multi_agent = True

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
agent_num = 8
collector_env_num = 8
evaluator_env_num = 8    # for debug use
n_episode = 8
num_simulations = 50
update_per_collect = 1000
batch_size = 256
reanalyze_ratio = 0.
action_space_size = 14
eps_greedy_exploration_in_collect = True

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================


smac_ez_config = dict(
    exp_name=f'data_ez_ctree/{env_name}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}',
    env=dict(
        env_name=env_name,
        map_name='3s5z',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        stop_value=0.999,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        multi_agent=multi_agent,
        ignore_done=True,
        model=dict(
            model_type='structure',
            agent_num=agent_num,
            obs_shape=150,
            global_obs_shape=216,
            action_shape=14,
            action_space_size=action_space_size,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        cuda=True,
        mcts_ctree=True,
        gumbel_algo=False,
        env_type='not_board_games',
        game_segment_length=500,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
        use_augmentation=False,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,
        ssl_loss_weight=0,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
    learn=dict(learner=dict(
        log_policy=True,
        hook=dict(log_show_after_iter=10, ),
    ), ),
)
smac_ez_config = EasyDict(smac_ez_config)
main_config = smac_ez_config

smac_ez_create_config = dict(
    env=dict(
        type='smac_lightzero',
        import_names=['zoo.smac.env.smac_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='multi_agent_efficientzero',
        import_names=['lzero.policy.multi_agent_efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
smac_ez_create_config = EasyDict(smac_ez_create_config)
create_config = smac_ez_create_config



if __name__ == "__main__":
    from zoo.smac.entry import train_muzero
    train_muzero((main_config, create_config), seed=0)