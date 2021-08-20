#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Task 1
#
print("Introduction to Python programming")

#
# Task 2
#
# TODO: Create a list with values 1--20.
list1To20 = list(range(1, 21))
print('List from 1 to 20: ', list1To20)

# TODO: Use a list comprehension to square all odd values in this list (keep even values unchanged).
listOddValuesSquared = []
for x in list1To20:
    if (x % 2 == 1): listOddValuesSquared.append(x*x)
print('Odd values squared: ', listOddValuesSquared)

# TODO: Request a number from the user (terminal keyboard input), loop this to get 4 numbers, and sort the numbers in ascending order.
listNum = []
for x in range(4):
    a = input('Enter a number: ')
    if (a.lstrip('-').isnumeric()): listNum.append(float(a))
print('sorted numbers: ', sorted(listNum))


#
# Task 3
#
# Write a function that ...
#
# TODO: ... squares all elements of a list.
def squareListElements(listArg):
    "returns a new list with all the numeric elements of the given list squared"
    listSquared = []
    for x in listArg:
        if (isinstance(x, int) or isinstance(x, float)): listSquared.append(x*x)
    return listSquared
print('square of list 1 to 20: ', squareListElements(list1To20))

# TODO: ... recursively calculates the sum of all elements in a list.
#def sumListElements(listArg):
#    "returns the sum of all numeric elements in the list"
#    sum = 0.0
#    for x in listArg:
#        if (isinstance(x, int) or isinstance(x, float)): sum = sum+x
#    return sum
#print('sum of list 1 to 20: ', sumListElements(list1To20))

def sumListElementsRecursively(listArg):
    "returns the sum of all numeric elements in the list"
    if len(listArg) == 0:
        return 0
    else:
        return listArg[0] + sumListElementsRecursively(listArg[1:])
print('sum of list 1 to 20 recursively: ', sumListElementsRecursively(list1To20))

# TODO: ... uses the built-in Python function `sum(list)` to calculate the arithmetic mean of all elements in a list.
def meanListElements(listArg):
    "returns the mean of all numeric elements in the list"
    try:
        return sum(listArg) / len(listArg)
    except Exception as e:
        print('This code NOT be running because of', e)
    return 0.0
print('mean of list 1 to 20: ', meanListElements(list1To20))

#
# Task 4
#

import math

# Write a class `Vec2` that has ...
class Vec2:
    # TODO: ... a variable `id`, and a "global" class variable `gid` that is used to assign each instance a unique `id`.
    __gid = 0

    # TODO: ... a constructor that initializes two variables `x` and `y`.
    def __init__(self, xArg, yArg):
        if not isinstance(xArg, int) or isinstance(yArg, float): raise TypeError("x and y must be integers")
        Vec2.__gid = Vec2.__gid + 1
        self.id = Vec2.__gid
        self.x = xArg
        self.y = yArg

    # TODO: ... a member function `length(self)` that calculates and returns the euclidean length of the vector.
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)
        # return math.dist([self.x], [self.y])

    # TODO: ... a member function `add(self, rhs)` that calculates the component-wise sum of the vector and another one (`rhs`)
    def add(self, rhs):
        if not isinstance(rhs, Vec2): raise TypeError("rhs must be of type Vec2")
        return Vec2(self.x + rhs.x, self.y + rhs.y)

    # To print the objects for Vec2 demo purposes
    def __str__(self):
        return "2-dimensional Vector (id: " + str(self.id) + ") ---> " + "x: " + str(self.x) + " y: " + str(self.y)


# TODO: Vec2 demo
obj1 = Vec2(2, 5)
print(obj1)
obj2 = Vec2(10, 26)
print(obj2)
obj3 = obj1.add(obj2)
print(obj3)
