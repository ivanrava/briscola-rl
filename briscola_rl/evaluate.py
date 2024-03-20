import logging

from tqdm import tqdm


def test_match(env, model, number_of_rounds: int = 10_000):
    logging.basicConfig(filename='test_match.log', level=logging.INFO)
    logger = logging.getLogger('test_match')
    total_reward = 0
    wins = 0
    draws = 0
    losses = 0
    for _ in tqdm(range(number_of_rounds)):
        reward = test_round(env, model, logger)
        total_reward += reward

        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1
    return (
        total_reward / number_of_rounds,
        wins / number_of_rounds,
        draws / number_of_rounds,
        losses / number_of_rounds
    )


def test_round(env, model, logger: logging.Logger) -> float:
    logger.info("Starting new round")
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, finisheds, _, info = env.step(action)
        total_reward += rewards
        if finisheds:
            logger.info("Finished round")
            return total_reward
