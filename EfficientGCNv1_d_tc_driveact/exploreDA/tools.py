import os
import numpy as np
import pickle

if __name__ == '__main__':

    path0 = "/home/bullet/PycharmProjects/LowlightRecognition/coldnight/n3_storage/data_advanced_splitrgb1_0316_softnode3/driveract/actionlevel/B_f90_0_2/train_label.pkl"
    path1 = "/home/bullet/PycharmProjects/LowlightRecognition/coldnight/n3_storage/data_advanced_splitrgb1_0316_softnode3/driveract/actionlevel/B_f90_0_2/val_label.pkl"
    path2 = "/home/bullet/PycharmProjects/LowlightRecognition/coldnight/n3_storage/data_advanced_splitrgb1_0316_softnode3/driveract/actionlevel/B_f90_0_2/test_label.pkl"

    with open(path0, 'rb') as f:
        samplename0, label0 = pickle.load(f, encoding='latin1')
    with open(path1, 'rb') as f:
        samplename1, label1 = pickle.load(f, encoding='latin1')
    with open(path2, 'rb') as f:
        samplename2, label2 = pickle.load(f, encoding='latin1')

    print(len(set(label0)), len(label0))
    print(len(set(label1)), len(label1))
    print(len(set(label2)), len(label2))

    combine_label = []
    combine_label.extend(list(label0))
    combine_label.extend(list(label1))
    combine_label.extend(list(label2))

    print(len(set(combine_label)), len(combine_label))

    label_ = list(set(combine_label))
    label_ = [int(x) for x in label_]
    label_.sort()
    print(label_)



"""
0	closing_door_outside  1
1	opening_door_outside  1
2	entering_car
3	closing_door_inside
4	fastening_seat_belt
5	using_multimedia_display
6	sitting_still
7	pressing_automation_button
8	fetching_an_object
9	opening_laptop
10	working_on_laptop
11	interacting_with_phone
12	closing_laptop
13	placing_an_object
14	unfastening_seat_belt
15	putting_on_jacket
16	opening_bottle
17	drinking
18	closing_bottle
19	looking_or_moving_around (e.g. searching)
20	preparing_food
21	eating
22	taking_off_sunglasses
23	putting_on_sunglasses
24	reading_newspaper
25	writing
26	talking_on_phone
27	reading_magazine
28	taking_off_jacket
29	opening_door_inside
30	exiting_car
31	opening_backpack
32	putting_laptop_into_backpack
33	taking_laptop_from_backpack



"""