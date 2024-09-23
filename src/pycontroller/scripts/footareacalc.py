root_w = 8
root_h = 12

tip_front = 1 # <= root_W
tip_back = 2 # >= root_W

extra_front = 1.5 # >= root_l
extra_back = 1.5 # >= root_W

root_A = root_w * root_h

front_A = (root_w + tip_front) * extra_front * 0.5
back_A = (root_w + tip_back) * extra_back * 0.5

total = front_A + root_A + back_A

print(total)
# total = ((root_w + tip_front) * extra_front * 0.5) + ((root_w + tip_back) * extra_back * 0.5) + (root_w * root_h)

