# Comparison Operators
# == equal to
# < less than
# <= less than or equal to
# != not equal to
# > greater than
# >= greater than or equal to
x = 5
y = 7
z = 15
print((x+5) >= y)

# Logical operations
# and &&
# not !
# or  ||
print((z>x) and (z<y))

# Conditional statement
if x > y:
    print("x is greater than y")
elif x == 5:
    print("x is equal to 5")
elif x < y:
    print("x is less than y")
else:
    print("x is equal to y")

# Looping statement
# while and for 
current_epoch = 1
num_epcohs = 100
while current_epoch <= num_epcohs:
    # print("Epoch " + str(current_epoch))
    # current_epoch = current_epoch + 1
    current_epoch += 1

# For Loop 
for current_epoch in range(0, 100, 2):
    print("Epoch " + str(current_epoch))

my_list = [0, 1, 2, 3, 4, 10]

for item in my_list:
    print(item)

i = 0
while i < len(my_list):
    item = my_list[i]
    print(item)
    i += 1

my_set = {0, 1, 2, 3, 4, 10}
for item in my_set:
    print(item)

classes = {"ship":1000, "plane":500}
for key, value in classes.items():
    print(key)
    print(value)

# List comprehension
my_range = [x**2 for x in range(0 ,100)]
# print(my_range)

mult_table = [x*y for x in range(1,10) for y in range(1, 10)]
# print(mult_table)

# Functions 
def ReLU(x: int):
    if x < 0:
        return 0
    else:
        return x

y = ReLU(100)
# y2 = ReLU("hello")
# print(y2)

def experiment(epochs, learning_rates: list, optim="SGD"):
    print("Running experiment using " + optim )
    lr_index = 0
    lr = learning_rates[lr_index]
    for epoch in range(0, epochs):
        if epoch % 25 == 0:
            lr = learning_rates[lr_index]
            lr_index += 1
        print(lr)

experiment(100, [0.1, 0.001, 0.0001, 0.00001], "ADAM")

# Recursion
def count_down(x: int):
    if x > 0:
        print(x)
        return count_down(x-1)
    else:
        print(x)

count_down(100)
