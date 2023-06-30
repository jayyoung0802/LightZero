from easydict import EasyDict

env_name = 'GoBigger'
multi_agent = True

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 32
n_episode = 32
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 2000
batch_size = 256
reanalyze_ratio = 0.
action_space_size = 27
direction_num = 12
eps_greedy_exploration_in_collect = True
random_collect_episode_num = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_efficientzero_config = dict(
    exp_name=
    f'data_ez_ctree/{env_name}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}',
    env=dict(
        env_name=env_name,
        team_num=2,
        player_num_per_team=2,
        direction_num=direction_num,
        step_mul=8,
        map_width=64,
        map_height=64,
        frame_limit=3600,
        action_space_size=action_space_size,
        use_action_mask=False,
        reward_div_value=0.1,
        reward_type='log_reward',
        contain_raw_obs=False,  # False on collect mode, True on eval vsbot mode, because bot need raw obs
        start_spirit_progress=0.2,
        end_spirit_progress=0.8,
        manager_settings=dict(
            food_manager=dict(
                num_init=260,
                num_min=260,
                num_max=300,
            ),
            thorns_manager=dict(
                num_init=3,
                num_min=3,
                num_max=4,
            ),
            player_manager=dict(ball_settings=dict(score_init=13000, ), ),
        ),
        playback_settings=dict(
            playback_type='by_frame',
            by_frame=dict(
                save_frame=False,  # when training should set as False
                save_dir='./',
                save_name_prefix='gobigger',
            ),
        ),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        multi_agent=multi_agent,
        model=dict(
            model_type='structured',
            latent_state_dim=176,
            frame_stack_num=1,
            action_space_size=action_space_size,
            downsample=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        cuda=True,
        mcts_ctree=True,
        gumbel_algo=False,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=random_collect_episode_num,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type='exp',
            start=1.,
            end=0.05,
            decay=int(1.5e4),
        ),
        use_augmentation=False,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,
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
atari_efficientzero_config = EasyDict(atari_efficientzero_config)
main_config = atari_efficientzero_config

atari_efficientzero_create_config = dict(
    env=dict(
        type='gobigger_lightzero',
        import_names=['zoo.gobigger.env.gobigger_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='gobigger_efficientzero',
        import_names=['lzero.policy.gobigger_efficientzero'],
    ),
    collector=dict(
        type='gobigger_episode_muzero',
        import_names=['lzero.worker.gobigger_muzero_collector'],
    )
)
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_gobigger
    train_muzero_gobigger([main_config, create_config], seed=seed)
