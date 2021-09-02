from gym.envs.registration import register


for reward_type in ["dense", "sparse"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {"reward_type": reward_type}

    for difficult_mode in ["easy", "hard"]:
        difficult_mode = "Hard" if difficult_mode == "hard" else ""
        kwargs["difficult_mode"] = difficult_mode

        register(
            id="ShadowHandReach{}-v1".format(suffix),
            entry_point="shadowhand_gym.envs:ShadowHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

    # Unset the difficult mode
    kwargs.pop("difficult_mode", None)

    register(
        id="ShadowHandBlock{}-v1".format(suffix),
        entry_point="shadowhand_gym.envs:ShadowHandBlockEnv",
        kwargs=kwargs,
        max_episode_steps=100,
    )
