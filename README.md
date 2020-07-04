# MMORPG Behavioral Data Anti-bot
Programs that automatize player actions (bots) have been one of the greatest problems in games, specially considering your use represents an unfair advantage towards other players. Many gaming companies choose to cope with bots because they create a fake volume of users but as stated by [Mishima 2013] this only leads to less and less real players.

This work was made as part of machine learning lectures and aims to use Python and scitkit-learn to try and detect bots through behavioral data.
It's made to be compatible with [The Forgotten Server (TFS)](https://github.com/otland/forgottenserver).

# Dataset used
By checking the code of some known bots, they usually aim to features that helps player evolution and minimize their losses. In a MMORPG such as TFS that would be auto healing when your life reaches a minimum threshold, collecting items automatically and quickly switching the target for fast combat. Knowing what is the goal of the botters, we could theorize about what data to collect in order to maximize our accuracy in detection.
One important hypothesis is that bots will show a distribution more mechanical than regular players, despite some bots trying to mitigate that including some noise, the behavioral curve seems to remain very separatable with a few transformations.

Since to actually check the results of our technique we would need a lot of data involved, what we did is collect a few data from real players and botters in different situations, analyze their projection and create functions that could generate new data using the same uniform normal distribution.
With the dataset generator in hands, we generate 1000 samples having 20% of it as botters. 
The following attributes are being considered as of now, all of them represents the values collected in a 5 minutes interval:
- collected_items: We assumed that because bots collect things faster, the amount of items would also play a certain role on it.
- avg_time_to_collect_item: Time in miliseconds between an item drops and gets collected. This can be deceived if the botter configures the noise to make actions with one second delay, for example.
- delta_time_to_collect_item: Standard deviation from avg_time_to_collect_item. This is one of the most significant measures to detect patterns and is a transformation quite simple to be done considering our 5 minutes interval.
- heal_threshold: the mean of current life percentage when a heal is performed. As vast majority of bots works with a heal threshold to trigger a instant heal, we are trying to understand what this threshold is configured.
- avg_time_to_start_healing: Mean of time in miliseconds between getting hit last and performing the heal. Bots are usually very quick here so we except very low values for botters. This can be deceived if the botter configures the noise to make actions with one second delay, for example.
- delta_time_to_start_healing: Since regular players tend to considerate the situation before healing, their standard deviation is quite high in comparison with something that is just applying patterns and executing what was configured.
- killed_enemies: We assumed that botters will try to play safe and usually stay away in areas where they have a low risk of dying, that implies killing weaker creatures and therefore having a higher kill count. Even if someone is in a regular area, within time they will certainly evolve, get stronger and start increasing the number too.
- hungry_time: More of a hipothesis that we wanted to check, players usually don't eat food in game but bots can be configured to always do so. Analyzing the hungry time may be a way to spotting a bot.

### How it works? (In Progress)
After generating 1000 samples with proportion 800 players / 200 botters, we run benchmark.py that will basically follow the procedure below:
- 

### Getting Started (In Progress)
- You need to make a few changes in your server to capture the necessary data. I'll soon be providing what are the editions and what's the improvements that have yet to be made in order to have it fully working in a crowded server.
- Depending on the data collected, it would be better if you run first the benchmark file in order to determine which is the best algorithm considering the data you have collected.
- If you don't want to go through the full analysis mode, just run antibot - decision tree version.py after dumping the 
