while True:
    target = [
        'OBJECT_TYPE_ALL_NS_LEVEL_2/LET-mAP: ',
        'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/LET-mAP: ',
        'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/LET-mAP: ',
        'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/LET-mAP: '
    ]
    out = []
    s = input()
    for t in target:
        l = s.find(t) + len(t)
        out.append(s[l:l+6])
    print("{} ({}/{}/{})".format(*out))

