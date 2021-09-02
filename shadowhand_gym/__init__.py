from gym.envs.registration import register


for reward_type in ["dense", "sparse"]:
    suffix = "Dense" if reward_type == "dense" else ""

    for difficult_mode in ["easy", "hard"]:
        suffix_2 = "Hard" if difficult_mode == "hard" else ""
        kwargs = {"reward_type": reward_type, "difficult_mode": difficult_mode}

        register(
            id="ShadowHandReach{}{}-v1".format(suffix, suffix_2),
            entry_point="shadowhand_gym.envs:ShadowHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

    # Difficult mode not needed for Block
    kwargs = {"reward_type": reward_type}

    register(
        id="ShadowHandBlock{}-v1".format(suffix),
        entry_point="shadowhand_gym.envs:ShadowHandBlockEnv",
        kwargs=kwargs,
        max_episode_steps=100,
    )
