
from CrossSection import CrossSectional
from utils import Newey_West ,eigen_risk_adj, eigenfactor_bias_stat, bayes_shrink,Newey_West_specific
from tqdm import tqdm
from util import *
from readdb import *
from data_gathering import get_data

class barra():
    def __init__(self,data, P, Q):
        '''
        :param data: base data plus industry factors data plus style factors data
        :param P: the industry nums
        :param Q: the style factor nums
        '''
        self.Q = Q                                                           # 风格因子数
        self.P = P                                                           # 行业因子数
        self.dates = pd.to_datetime(data.date.values)                        # 日期
        self.sorted_dates = pd.to_datetime(np.sort(pd.unique(self.dates)))   # 排序后的日期
        self.T = len(self.sorted_dates)                                      # 期数
        self.data = data                                                     # 数据
        self.columns = ['country']                                           # 因子名
        self.columns.extend((list(data.columns[4:])))

        self.last_capital = None
        self.factor_ret = None
        self.specific_ret = None
        self.R2 = None

        self.Newey_West_cov = None

        self.eigen_risk_adj_cov = None
        self.vol_regime_adj_cov = None



    def reg_by_time(self):
        '''
        using cross section to get the factor return
        :return:
        '''
        factor_ret = []
        R2 = []

        specific_ret = []
        print('=='*10 +'逐时间点做横截面的因子的回归'+"=="*10)
        for t in range(self.T):
            data_by_time = self.data.iloc[self.dates == self.sorted_dates[t], :]
            data_by_time = data_by_time.sort_values(by='code')

            base,style,indus = data_by_time.iloc[:, :4],data_by_time.iloc[:, -self.Q:],data_by_time.iloc[:, 4:(4 + self.P)]
            cs = CrossSectional(base, style, indus)
            factor_ret_t, specific_ret_t, _, R2_t = cs.reg()
            factor_ret.append(factor_ret_t)
            # 注意：每个截面上股票池可能不同
            specific_ret.append(pd.DataFrame([specific_ret_t], columns=cs.stocknames, index=[self.sorted_dates[t]]))
            R2.append(R2_t)
            self.last_capital = cs.capital


        factor_ret = pd.DataFrame(factor_ret, columns = self.columns, index = self.sorted_dates)
        R2 = pd.DataFrame(R2, columns = ['R2'], index = self.sorted_dates)
        self.factor_ret = factor_ret  # 因子收益
        self.specific_ret = specific_ret  # 特异性收益
        self.R2 = R2  # R2
        return ((factor_ret, specific_ret, R2))

    def Newey_West_by_time(self,q=2,tao = 90):

        if self.factor_ret is None:
            raise Exception('please run reg_by_time to get factor returns first')

        Newey_West_cov = []
        print('\n\n===================================逐时间点进行Newey West调整=================================')
        for t in tqdm(range(1, self.T + 1)):
            try:
                Newey_West_cov.append(Newey_West(self.factor_ret[:t], q, tao))
            except:
                Newey_West_cov.append(pd.DataFrame())


        self.Newey_West_cov = Newey_West_cov
        return (Newey_West_cov)


    def eigen_risk_adj_by_time(self,M = 100, scale_coef = 1.4):
        '''
        逐时间点进行Eigenfactor Risk Adjustment
        M: 模拟次数
        scale_coef: scale coefficient for bias
        '''

        if self.Newey_West_cov is None:
            raise Exception(
                'please run Newey_West_by_time to get factor return covariances after Newey West adjustment first')

        eigen_risk_adj_cov = []
        print('\n\n===================================逐时间点进行Eigenfactor Risk调整=================================')
        for t in tqdm(range(self.T)):
            try:
                eigen_risk_adj_cov.append(eigen_risk_adj(self.Newey_West_cov[t], self.T, M, scale_coef))
            except:
                eigen_risk_adj_cov.append(pd.DataFrame())

        self.eigen_risk_adj_cov = eigen_risk_adj_cov
        return (eigen_risk_adj_cov)

    def vol_regime_adj_by_time(self,tao = 84):
        '''
               Volatility Regime Adjustment
               tao: Volatility Regime Adjustment的半衰期
        '''

        if self.eigen_risk_adj_cov is None:
            raise Exception(
                'please run eigen_risk_adj_by_time to get factor return covariances after eigenfactor risk adjustment first')

        K = len(self.eigen_risk_adj_cov[-1])
        factor_var = list()
        for t in tqdm(range(self.T)):
            factor_var_i = np.diag(self.eigen_risk_adj_cov[t])
            if len(factor_var_i) == 0:
                factor_var_i = np.array(K * [np.nan])
            factor_var.append(factor_var_i)

        factor_var = np.array(factor_var)
        B = np.sqrt(np.mean(self.factor_ret ** 2 / factor_var, axis=1))  # 截面上的bias统计量
        weights = 0.5 ** (np.arange(self.T - 1, -1, -1) / tao)  # 指数衰减权重

        lamb = []
        vol_regime_adj_cov = []
        print('\n\n==================================逐时间点进行Volatility Regime调整================================')
        for t in tqdm(range(1, self.T + 1)):
            # 取除无效的行
            okidx = pd.isna(factor_var[:t]).sum(axis=1) == 0
            okweights = weights[:t][okidx] / sum(weights[:t][okidx])
            fvm = np.sqrt(sum(okweights * B.values[:t][okidx] ** 2))  # factor volatility multiplier

            lamb.append(fvm)
            vol_regime_adj_cov.append(self.eigen_risk_adj_cov[t - 1] * fvm ** 2)

        self.vol_regime_adj_cov = vol_regime_adj_cov
        return ((vol_regime_adj_cov, lamb))


if __name__ == '__main__':

    # import os
    # os.chdir('C:\\Users\\asus\\Desktop\\MFM')

    # data,nums_indus,factor_nums = get_data(start='2012-02-01',end='2018-06-01',store=True)
    data = pd.read_csv('barra_data.csv')
    #
    data = data.dropna(axis=0)


    model = barra(data, 28, 10)
    # model = barra(data, nums_indus, factor_nums)
    (factor_ret, specific_ret, R2) = model.reg_by_time()
    # print(specific_ret)
    nw_cov_ls = model.Newey_West_by_time(q = 2, tao = 90)                         #Newey_West调整

    er_cov_ls = model.eigen_risk_adj_by_time(M = 100, scale_coef = 1.4)           #特征风险调整
    vr_cov_ls, lamb = model.vol_regime_adj_by_time(tao = 90)                      #vol regime调整

    eigenfactor_bias_stat(nw_cov_ls[1000:], factor_ret[1000:], predlen = 21)      #特征风险调整前特征因子组合的bias统计量
    # eigenfactor_bias_stat(er_cov_ls[1000:], factor_ret[1000:], predlen = 21)    #特征风险调整后特征因子组合的bias统计量

    # 假如 截面上的股票都是一样的 就可以使用下面对于特异性收益的处理
    a = []
    for i in specific_ret:
        a.append(list(i.values)[0])
    specific = pd.DataFrame(a)     # 未定义columns

    vr_cov_sepcific = Newey_West_specific(specific, q=2, tao=90)

    capital = data.groupby('date').get_group(data.date.values[-1]).captial
    group = pd.qcut(capital.values, 10).codes
    volatility = np.diag(vr_cov_sepcific)
    bayes_specific = bayes_shrink(volatility, capital.values, ngroup=10, q=1)



