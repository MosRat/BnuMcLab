from utils import *
from data_deal import *
from decision import *
from plot import *

if __name__ == '__main__':
    # 载入数据
    d1, d2, d3 = load_dataset()

    # 创建分类器
    handle = BayesMinErrorDecision()

    # 拟合,可选参数d1,d2,d3
    handle.fit(d3)

    # 打印拟合（最大似然估计）的参数
    print(handle)

    # 预测
    print(f'模型预测值 ：\n{handle.predict()}')

    # 评分
    print(f'模型得分（正确率）：{handle.score()}')

    # 绘制混淆矩阵
    plot_matrix(handle.confuse_matrix())

    # 计算F1 Score
    print(f"模型f1得分 :{handle.f1_score()}")

    # 获取边界
    # edge = handle.get_dec_edge(np.linspace(-6, 8, 1000), np.linspace(-6, 8, 1000))

    # 绘制等概率密度线以及决策面
    # fig = plt.figure(dpi=150)
    # ax = plot_2d(fig, 2)
    # ax.plot(edge[:, 0], edge[:, 1], color='#fdbf6f')
    # plt.show()

    # 绘制散点图以及决策面
    # fig = plt.figure(dpi=300, figsize=(8, 8))
    # plot_scatter(plt.gca(), handle.X_train, handle.y_train,
    #              handle.get_dec_edge(np.linspace(-6, 8, 500), np.linspace(-6, 8, 500)))
    # plt.show()

    # 绘制概率分布密度图和决策面
    # fig = plt.figure(dpi=160, figsize=(8, 8))
    # ax = fig.add_subplot(int(f'111'), projection='3d')
    # x, y = edge[:, 0], edge[:, 1]
    # plot_3d_edge(ax, np.concatenate([x, y], axis=1))
    #
    # X, Y, Z = make_gussin_data(params_mean[0], params_cov[0])
    # plot_dis_3d(ax, X, Y, Z)
    #
    # X, Y, Z = make_gussin_data(params_mean[1], params_cov[1])
    # plot_dis_3d(ax, X, Y, Z)
    # plt.show()
