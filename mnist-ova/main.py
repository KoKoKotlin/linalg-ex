from mnist import MNIST
from perceptron import LinearPerceptron
from random import choices

BADGE_SIZE = 100

def main():
    images, labels = MNIST("train").load_training()
    data = list(zip(images, labels))

    zero = LinearPerceptron(len(images[0]))
    for x, y in (badge := choices(data, k=BADGE_SIZE)):
        y = -1 if not y else 1
        zero.train(x, y)
    print(zero.weights)

if __name__ == "__main__":
    main()