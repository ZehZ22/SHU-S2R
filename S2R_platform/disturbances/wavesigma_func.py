# wavesigma_func.py

def wavesigma_func(omega, T1):
    """
    高频波浪谱的标准差计算函数
    :param omega: 角频率 (rad/s)
    :param T1: 平均周期 (s)
    :return: sigma
    """
    return 0.5 * omega * T1
