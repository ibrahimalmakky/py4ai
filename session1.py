z = 19 % 5
# print(z)

exp = 2 ** 4
# print(exp)

fl = 10 // 3
# print(fl)

# 0010
# 0011
# 0010
bitw = 2 & 3
# print(bitw)

# Lists
my_list = [1, 4, 4.6, "mystring"]
# print(my_list[3])
# print(len(my_list))
# my_list.append(4)
# Change value in the list
my_list[1] = 9
# print(my_list)
print(my_list[-1:-4:-1])

# Tuples
my_tuple = (5,3)
# This will give an error
# my_tuple[0] = 7
# print(my_tuple[1])
coordinates = (24.432606315672004, 54.61841348127513)

# Sets
my_set = {5, 2, 6.7, 5}
my_pop = my_set.pop()
print(my_pop)

# Dictionaries
my_dict = {"Abu Dhabi": (24.432606315672004, 54.61841348127513),
            "Zoo": 54,
            "my_list": [7, 3, 7],
            "my_other_dict": {},
            8: 5}
my_dict["Dubai"] = (24.432606315672004, 56.61841348127513)
print(my_dict["my_list"][1])

str1 = "Hello"
# str1[0] = "Z"
str2 = " World"
conc_str = str1 + str2
print(conc_str[0:5])

my_quote = 'this is the quote: ""'
mult_line_str = """ 
Example
Multi-line
String
"""

learning_rate = input("Please enter the learning rate")
learning_rate = float(learning_rate)
test = learning_rate * 2
print(test)
