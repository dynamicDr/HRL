import pandas as pd
from matplotlib import pyplot as plt


def match_plot(df, folder=None):
    df_column_names = ["match", "blue_score", "yellow_score", "blue_robot_0_possession_frame",
                       "blue_robot_1_possession_frame", "blue_robot_2_possession_frame",
                       "yellow_robot_0_possession_frame",
                       "yellow_robot_1_possession_frame", "yellow_robot_2_possession_frame",
                       "blue_robot_0_intercept_time", "blue_robot_1_intercept_time", "blue_robot_2_intercept_time",
                       "yellow_robot_0_intercept_time", "yellow_robot_1_intercept_time",
                       "yellow_robot_2_intercept_time",
                       "blue_robot_0_pass_time", "blue_robot_1_pass_time", "blue_robot_2_pass_time",
                       "yellow_robot_0_pass_time", "yellow_robot_1_pass_time", "yellow_robot_2_pass_time",
                       "ball_in_blue_half_frame", "ball_in_yellow_half_frame"]

    plt.plot(df["match"], df["blue_score"], color="Blue")
    plt.plot(df["match"], df["yellow_score"], color="Orange")
    plt.legend(["Blue Team Score", "Yellow Team Score"])
    if folder is not None:
        plt.savefig(f"{folder}/goal")
    plt.show()
    plt.clf()

    goal_diff = df["blue_score"] - df["yellow_score"]
    # plt.plot(df["match"],[0 for i in range(len(df["match"]))],color="Black")
    plt.plot(df["match"], goal_diff, color="Red")
    # plt.axis('off')
    plt.axhline(0, color="Black", linewidth=1)
    plt.legend(["Goal difference"])
    if folder is not None:
        plt.savefig(f"{folder}/goal_difference")
    plt.show()
    plt.clf()

    plt.plot(df["match"], df["ball_in_blue_half_frame"], color="Blue")
    plt.plot(df["match"], df["ball_in_yellow_half_frame"], color="Orange")
    plt.legend(["ball_in_blue_half_frame", "ball_in_yellow_half_frame"])
    if folder is not None:
        plt.savefig(f"{folder}/ball_position")
    plt.show()
    plt.clf()

    robot_colors = ["blue", "navy", "royalblue", "orange", "yellow", "gold"]
    for i in range(6):
        plt.plot(df["match"], df[df_column_names[3 + i]], color=robot_colors[i])
    plt.legend(df_column_names[3:3 + 6])
    if folder is not None:
        plt.savefig(f"{folder}/possession")
    plt.show()
    plt.clf()

    for i in range(6):
        plt.plot(df["match"], df[df_column_names[9 + i]], color=robot_colors[i])
    plt.legend(df_column_names[9:9 + 6])
    if folder is not None:
        plt.savefig(f"{folder}/intercept")
    plt.show()
    plt.clf()

    for i in range(6):
        plt.plot(df["match"], df[df_column_names[15 + i]], color=robot_colors[i])
    plt.legend(df_column_names[15:15 + 6])
    if folder is not None:
        plt.savefig(f"{folder}/passing")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    number = 24
    df = pd.read_csv(f"/home/user/football/HRL/match_result/25_2022-09-08 14:56:24.969044/25.csv")
    match_plot(df)
