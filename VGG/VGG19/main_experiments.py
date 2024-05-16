from ..implementation import main_experiment

# Repeat the experiment 4 times to maintain reliable results

for i in range(1, 5):
    main_experiment(i, 'VGG19')