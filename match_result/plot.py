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
                       "ball_in_blue_half_frame", "ball_in_yellow_half_frame","coach_top_1_acc","coach_top_2_acc"]

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

    df['possession_blue'] = df[df_column_names[3]] + df[df_column_names[4]]+ df[df_column_names[5]]
    df['possession_yellow'] = df[df_column_names[6]] + df[df_column_names[7]]+ df[df_column_names[8]]
    plt.plot(df["match"], df["possession_blue"], color="Blue")
    plt.plot(df["match"], df["possession_yellow"], color="Yellow")
    plt.legend(["possession_blue", "possession_yellow"])
    if folder is not None:
        plt.savefig(f"{folder}/possession_sum")
    plt.show()
    plt.clf()

    for i in range(6):
        plt.plot(df["match"], df[df_column_names[9 + i]], color=robot_colors[i])
    plt.legend(df_column_names[9:9 + 6])
    if folder is not None:
        plt.savefig(f"{folder}/intercept")
    plt.show()
    plt.clf()

    df['intercept_blue'] = df[df_column_names[9]] + df[df_column_names[10]] + df[df_column_names[11]]
    df['intercept_yellow'] = df[df_column_names[12]] + df[df_column_names[13]] + df[df_column_names[14]]
    plt.plot(df["match"], df["intercept_blue"], color="Blue")
    plt.plot(df["match"], df["intercept_yellow"], color="Yellow")
    plt.legend(["intercept_blue", "intercept_yellow"])
    if folder is not None:
        plt.savefig(f"{folder}/intercept_sum")
    plt.show()
    plt.clf()

    for i in range(6):
        plt.plot(df["match"], df[df_column_names[15 + i]], color=robot_colors[i])
    plt.legend(df_column_names[15:15 + 6])
    if folder is not None:
        plt.savefig(f"{folder}/passing")
    plt.show()
    plt.clf()

    df['passing_blue'] = df[df_column_names[15]] + df[df_column_names[16]] + df[df_column_names[17]]
    df['passing_yellow'] = df[df_column_names[18]] + df[df_column_names[19]] + df[df_column_names[20]]
    plt.plot(df["match"], df["passing_blue"], color="Blue")
    plt.plot(df["match"], df["passing_yellow"], color="Yellow")
    plt.legend(["passing_blue", "passing_yellow"])
    if folder is not None:
        plt.savefig(f"{folder}/passing_sum")
    plt.show()
    plt.clf()

    plt.plot(df["match"], df["coach_top_1_acc"], color="Blue")
    plt.plot(df["match"], df["coach_top_2_acc"], color="Red")
    plt.legend(["coach_top_1_acc", "coach_top_2_acc"])
    if folder is not None:
        plt.savefig(f"{folder}/caoch_acc")
    plt.show()
    plt.clf()

if __name__ == '__main__':
    number = 24
    df = pd.read_csv(f"/home/user/football/HRL/match_result/25_2022-09-08 14:56:24.969044/25.csv")
    match_plot(df)
