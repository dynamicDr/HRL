import numpy as np
robots_blue_x = [2,0,1]
robots_blue_y = [0,3,1]
def _closet_move():
    robots_distance_to_ball = [[], []]
    for idx in range(3):
        ball = np.array([0, 0])
        robot = np.array([robots_blue_x[idx],
                          robots_blue_y[idx]])
        robot_distance_to_ball = np.sqrt(sum((robot - ball) ** 2 for robot, ball in zip(robot, ball)))
        robots_distance_to_ball[0].append(idx)
        robots_distance_to_ball[1].append(robot_distance_to_ball)
    min_idx = robots_distance_to_ball[0][np.argmin(robots_distance_to_ball[1])]
    return min_idx

if __name__ == '__main__':
    #print(_closet_move())
    ball = np.array([1, -2])
    #print(np.linalg.norm(ball))
    a = np.array([1,-2.2])
    b = np.array([3,4])
    #print(-np.dot(a, b))
    print(np.abs(a[1]))
