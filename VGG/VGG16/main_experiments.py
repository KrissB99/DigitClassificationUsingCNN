from ..implementation import main_experiment

# Repeat the experiment 4 times to maintain reliable results

for x in range(1, 5):
    main_experiment(i=x, vgg_model_type='VGG16', epochs=10, size=50000)