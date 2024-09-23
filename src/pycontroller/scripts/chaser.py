enabled = False

state = 0

def loop(tilt, pan, walking, isLost = False):
    if not isLost:
        walking.vectorTarget.x = pan