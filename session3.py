# Class
class Experiment:

    optims = ["SGD", "ADAM"]

    def __init__(self, epochs, learning_rate: int, optim="SGD"):
        self.epochs = epochs
        self.lr = learning_rate
        self.optim = optim

    def train(self):
        print("training..")
        for epoch in range(0, self.epochs):
            print("epoch " + str(epoch) + ": lr=" + str(self.lr))
            self.lr_decay()

    def lr_decay(self):
        self.lr -= 0.01

# Inheritance
class VisionExp(Experiment):

    def __init__(self, epochs, learning_rate: int):
        super().__init__(epochs, learning_rate, "ADAM")

    # Polymorphism (Method overloading)
    def lr_decay(self):
        self.lr -= 0.001

# new_experiment = Experiment(10, 0.1)
# new_experiment.train()
# my_vision_exp = VisionExp(8, 0.5)
# my_vision_exp.train()


