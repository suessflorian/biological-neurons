
#!/bin/bash
set -x

# Infinite loop, progress while I'm not around...
while true; do
    for model in VGG11 VGG13 VGG16 VGG19; do
        echo "Training $model"
        python -m main.py --lr 0.01 --model $model --resume --variant LIF
    done

    for model in VGG11 VGG13 VGG16 VGG19; do
        echo "Training $model"
        python -m main.py --lr 0.01 --model $model --resume
    done

    for model in LeNet LeNet5; do
        echo "Training $model"
        python -m main.py --lr 0.01 --model $model --resume --variant LIF
    done

    for model in LeNet LeNet5; do
        echo "Training $model"
        python -m main.py --lr 0.01 --model $model --resume
    done
done
