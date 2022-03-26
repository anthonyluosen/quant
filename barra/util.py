from readdb import *
import statsmodels.api as sm
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime,timedelta
from scipy import stats

def get_trading_day_list(start_time, end_time, frequency=None):
    """
    获取交易日期列表
    input:
    start:str or datetime,起始时间
    end:str or datetime，终止时间
    frequency:
        str: day,month,quarter,halfyear,默认为day
        int:间隔天数
    """
    if isinstance(frequency, int):
        all_trade_days = get_trade_days(start_time, end_time)
        trade_days = all_trade_days[::frequency]
        return trade_days
    df = get_index_price('000300.XSHG', start_date=start_time, end_date=end_time)
    df.set_index("date", inplace=True)
    if not frequency or frequency == 'day':
        days = df.index
    else:
        df['year-month'] = [str(i)[0:7] for i in df.index]
        days = []
        if frequency == 'month':
            days = df.drop_duplicates('year-month').index
        elif frequency == 'quarter':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month'] == '01') | (df['month'] == '04') | (df['month'] == '07') | (df['month'] == '10')]
            days = df.drop_duplicates('year-month').index
        elif frequency == 'halfyear':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month'] == '01') | (df['month'] == '07')]
            days = df.drop_duplicates('year-month').index
    trade_days = [pd.to_datetime(i).strftime('%Y-%m-%d') for i in days]
    return trade_days


def date_delay(date_list,ndays):
    '''
    将date_list转化为ndays天前的日期list["%Y-%m-%d","%Y-%m-%d",....]
    :param date_list: list 日期（str）
    :param ndays: int
    :return: list
    '''
    tmp = []
    for da in date_list:
        str2date = datetime.strptime(da, "%Y-%m-%d")
        date = str2date - timedelta(days=ndays)
        str = date.strftime('%Y-%m-%d')
        tmp.append(str)
    return tmp

def get_periods_index_stocks(index, start_date, end_date=False):
    """
    获取一段时间股票的组成股票代码
    :param index:  指数代码
    :param start_date: 开始日期
    :param end_date:  截至日期
    :return:
    """
    assert index, "index_symbol is required"
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_month_index"]
    if end_date:
        myquery = ({"index_code":  index, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
        mydoc = mycol.find(myquery)
        data = pd.DataFrame(list(mydoc))
        data = data["code"].unique().tolist()
    else:
        myquery = ({"index_code":  index, "date": {"$gte": parse(date_delay([start_date],40)[0]), "$lte": parse(start_date)}})
        mydoc = mycol.find(myquery)
        data = pd.DataFrame(list(mydoc))
        data = data[data['date'] == data['date'].iloc[-1]]
        data = data["code"].unique().tolist()
    return data




def get_stock_industry(stocks, industry_type, date):
    """
    获取股票代码
    :param stocks: 股票代码
    :param industry_type: str, 行业代码
    "sw_l1": 申万一级行业
    "sw_l2": 申万二级行业
    "sw_l3": 申万三级行业
    "jq_l1": 聚宽一级行业
    "jq_l2": 聚宽二级行业
    "zjw": 证监会行业
    :param date:  日期
    :return: series index为股票代码，值为行业代码
    """
    assert industry_type, "industry_code is required"
    if isinstance(stocks, str):
        stocks = (stocks,) * 2
    elif isinstance(stocks, list):
        stocks = tuple(stocks)
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_daily_industry"]
    myquery = ({"code": {"$in": stocks}, "date":  parse(date)})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc))
    data = pd.Series(data[industry_type].values, index=data['code'])
    return data


def fillna_with_industry(data, date, industry_name='sw_l1'):
    """
    使用行业均值填充nan值
    input:
    data：DataFrame,输入数据，index为股票代码
    date:string,时间必须和data数值对应时间一致
    output：
    DataFrame,缺失值用行业中值填充，无行业数据的用列均值填充
    """
    if isinstance(data, pd.Series):
        data = data.to_frame(name='factor')
    stocks = list(data.index)
    stocks_industry = get_stock_industry(stocks, industry_name, date)
    stocks_industry = stocks_industry.to_frame(name='industry')
    stocks_industry_merge = data.merge(stocks_industry, left_index=True, right_index=True, how='left')
    columns = list(data.columns)
    select_data = []
    group_data = stocks_industry_merge.groupby('industry')
    group_data_mean = group_data.mean()
    group_data = stocks_industry_merge.merge(group_data_mean, left_on='industry', right_index=True, how='left')
    for column in columns:
        group_data.loc[pd.isnull(group_data[column + '_x']),column + '_x'] = group_data.loc[pd.isnull(group_data[column + '_x']),column + '_y']
        group_data[column] = group_data[column + '_x']
        select_data.append(group_data[column])
    result = pd.concat(select_data, axis=1)
    # 行业均值为Nan,用总体均值填充
    mean = result.mean()
    for i in result.columns:
        result[i].fillna(mean[i], inplace=True)
    return result


def fillna_industry(data, stocks_industry):
    """
    使用行业均值填充nan值
    input:
    data：series,输入数据，index为股票代码,value因子值
    stocks_industry：series,输入数据，index为股票代码,value行业代码
    output：
    DataFrame,缺失值用行业中值填充，无行业数据的用列均值填充
    """
    if isinstance(data, pd.Series):
        data = data.to_frame(name='factor')
    if isinstance(stocks_industry, pd.Series):
        stocks_industry = stocks_industry.to_frame(name='industry')
    stocks_industry_merge = data.merge(stocks_industry, left_index=True, right_index=True, how='left')
    columns = list(data.columns)
    col = stocks_industry.columns[0]
    select_data = []
    group_data = stocks_industry_merge.groupby(col)
    group_data_mean = group_data.mean()
    group_data = stocks_industry_merge.merge(group_data_mean, left_on=col, right_index=True, how='left')
    for column in columns:
        group_data.loc[pd.isnull(group_data[column + '_x']),column + '_x'] = group_data.loc[pd.isnull(group_data[column + '_x']),column + '_y']
        group_data[column] = group_data[column + '_x']
        select_data.append(group_data[column])
    result = pd.concat(select_data, axis=1)
    # 行业均值为Nan,用总体均值填充
    mean = result.mean()
    for i in result.columns:
        result[i].fillna(mean[i], inplace=True)
    return result


def filter_extreme_MAD(factor, n=3):
    """
     MAD: 中位数去极值
    :param factor: series index为股票代码 value为因子值
    :param n:
    :return:
    """
    median = factor.quantile(0.5)
    new_median = ((factor - median).abs()).quantile(0.50)
    max_range = median + n * new_median
    min_range = median - n * new_median
    if isinstance(factor, pd.DataFrame):
        return np.clip(factor, min_range, max_range,axis=1)
    if isinstance(factor, pd.Series):
        return np.clip(factor, min_range, max_range)

def standardize(factor, ty=2):
    """
     标准化函数：
    :param factor: series index为股票代码 value为因子值
    ty为标准化类型:1 MinMax,2 Standard,3 maxabs
    """
    data = factor.copy()
    if int(ty) == 1:
        ret = (data - data.min()) / (data.max() - data.min())
    elif ty == 2:
        ret = (data - data.mean()) / data.std()
    elif ty == 3:
        ret = data / 10 ** np.ceil(np.log10(data.abs().max()))
    else:
        ret = None
    return ret


def neutralize(data, date, market_cap=None, industry_name='sw_l1', country_factor=False):
    """
    中性化，使用行业和市值因子中性化
    input：
    data：DataFrame,index为股票代码，columns为因子，values为因子值
    name:str,行业代码
    "sw_l1": 申万一级行业
    "sw_l2": 申万二级行业
    "sw_l3": 申万三级行业
    "jq_l1": 聚宽一级行业
    "jq_l2": 聚宽二级行业
    "zjw": 证监会行业
    date:获取行业数据的时间
    market_cap:市值因子
    """
    if isinstance(data, pd.Series):
        data = data.to_frame(name='factor')
    stocks = data.index.unique().tolist()
    data1 = data.copy()
    index = list(data.index)
    if industry_name is None:
        if isinstance(market_cap, pd.Series):
            market_cap = market_cap.to_frame()
            market_cap = np.log(market_cap.loc[index])
            x = market_cap
    else:
        data1['se'] = get_stock_industry(stocks, industry_name, date)
        industry_se = data1['se']
        industry_se = np.array(industry_se.loc[index].tolist())
        industry_dummy = sm.categorical(industry_se, drop=True)
        industry_dummy = pd.DataFrame(industry_dummy, index=index)
        if market_cap is None:
            x = industry_dummy
        else:
            if isinstance(market_cap, pd.Series):
                market_cap = market_cap.to_frame()
            market_cap = np.log(market_cap.loc[index])
            x = pd.concat([industry_dummy, market_cap], axis=1)
    if country_factor:
        x = sm.add_constant(x)
    model = sm.OLS(data, x)
    result = model.fit()
    y_fitted = result.fittedvalues
    col = data.columns.values[0]
    neu_result = data - y_fitted.to_frame(name=col)
    return neu_result


def get_profit_depend_timelist(stocks, timelist, month_num=1, cal_num=1):
    """
    input:
    stocks:list 股票代码
    timelist: 时间序列
    month_num:计算几个月的收益率，默认为1，即一个月的收益率
    cal_num:int，计算每月最后n天的收盘价均值，默认为1
    output：
    DataFrame, index为code，column为日期，值为收益率
    """
    price_list = []
    for date in timelist:
        price = get_price(stocks, count=cal_num, end_date=date)
        price = pd.pivot_table(price, index="date", columns="code", values="close")
        price_mean = price.mean().to_frame()
        price_mean.columns = [date]
        price_list.append(price_mean)
    profit = pd.concat(price_list, axis=1)
    profit_pct = profit.pct_change(month_num, axis=1).dropna(axis=1, how='all')
    return profit_pct


def get_group_profit(factor, group,startdate,enddate,market=True,marketcap=None,cal_num=1):
    """ 计算一个周期内的分组收益率
    input:
    factor:dataframe 因子已排序 index为code
    group: 分组数
    startdate: 开始日期
    enddate: 结束日期
    market:市值加权为True，False等权
    market_cap:若市值加权则输入，index为code，值为市值
    cal_num:int，计算每月最后n天的收盘价均值，默认为1
    output：
    profit： dict, key为组号，value为收益率
    code： dict, key为组号，value为股票代码list
    """
    num = math.ceil(len(factor) / group) #每组股票数量
    profit = {}
    sort_list = factor.index.tolist()
    code={}
    if market:
        if isinstance(marketcap, pd.Series):
            marketcap = marketcap.to_frame(name='market')
        tmp_profit = get_profit_depend_timelist(factor.index.values.tolist(), [startdate, enddate], month_num=1,
                                                cal_num=cal_num)
        tmp_profit = tmp_profit.loc[sort_list]
        col = tmp_profit.columns[0]
        col1 = marketcap.columns[0]
        tmp_profit = pd.merge(tmp_profit, marketcap, left_index=True, right_index=True)
        tmp_profit['value'] = tmp_profit[col] * tmp_profit[col1]
        for m in range(0, group - 1):
            profit[m] = tmp_profit['value'][m * num:num + m * num].sum()/tmp_profit[col1][m * num:num + m * num].sum()
            code[m] = tmp_profit.index[m * num:num + m * num].tolist()
        profit[m + 1] = tmp_profit['value'][(m + 1) * num:].sum()/tmp_profit[col1][(m + 1) * num:].sum()
        code[m+1] = tmp_profit.index[(m + 1) * num:].tolist()
        pass
    else:
        tmp_profit = get_profit_depend_timelist(factor.index.values.tolist(), [startdate, enddate], month_num=1,
                                                cal_num=cal_num)
        tmp_profit = tmp_profit.loc[sort_list]
        for m in range(0, group - 1):
            profit[m] = tmp_profit[m * num:num + m * num].mean()[0]
            code[m] = tmp_profit.index[m * num:num + m * num].tolist()
        profit[m + 1] = tmp_profit[(m + 1) * num:].mean()[0]
        code[m + 1] = tmp_profit.index[(m + 1) * num:].tolist()
    return profit,code


def group_profit(factor,tmp_profit,group,market=True,marketcap=None):
    """ 计算一个周期内的分组收益率
    input:
    factor:dataframe 因子已排序 index为code
    tmp_profit:dataframe 因子收益率 index为code
    group: 分组数
    market:市值加权为True，False等权
    market_cap:若市值加权则输入，index为code，值为市值
    output：
    profit： dict, key为组号，value为收益率
    code： dict, key为组号，value为股票代码list
    """
    num = math.ceil(len(factor) / group) #每组股票数量
    profit = {}
    sort_list = factor.index.tolist()
    code = {}
    if isinstance(tmp_profit, pd.Series):
        tmp_profit = tmp_profit.to_frame(name='profit')
    if market:
        if isinstance(marketcap, pd.Series):
            marketcap = marketcap.to_frame(name='market')
        tmp_profit = tmp_profit.loc[sort_list]
        col = tmp_profit.columns[0]
        col1 = marketcap.columns[0]
        tmp_profit = pd.merge(tmp_profit, marketcap, left_index=True, right_index=True)
        tmp_profit['value'] = tmp_profit[col] * tmp_profit[col1]
        for m in range(0, group - 1):
            profit[m] = tmp_profit['value'][m * num:num + m * num].sum()/tmp_profit[col1][m * num:num + m * num].sum()
            code[m] = tmp_profit.index[m * num:num + m * num].tolist()
        profit[m + 1] = tmp_profit['value'][(m + 1) * num:].sum()/tmp_profit[col1][(m + 1) * num:].sum()
        code[m+1] = tmp_profit.index[(m + 1) * num:].tolist()
        pass
    else:
        tmp_profit = tmp_profit.loc[sort_list]
        for m in range(0, group - 1):
            profit[m] = tmp_profit[m * num:num + m * num].mean()[0]
            code[m] = tmp_profit.index[m * num:num + m * num].tolist()
        profit[m + 1] = tmp_profit[(m + 1) * num:].mean()[0]
        code[m + 1] = tmp_profit.index[(m + 1) * num:].tolist()
    return profit,code


def profit_2df(profit):
    '''
    字典comprofit转DataFrame
    :param profit:
    :return: DataFrame
    '''
    tmp_list = []
    for key, value in profit.items():
        tmp_list.append((pd.DataFrame(value, index=[key]) + 1))
    return pd.concat(tmp_list, axis=0)


def mkdir(path):
    '''
    创建文件夹
    '''
    exist=os.path.exists(path)
    if not exist:
        os.makedirs(path)
    return path


def cut_group(result,mc,ggg=1):
    '''
    依照市值进行分组
    :param result: factor,index=code
    :param mc: marketcap,index=code
    :return: [Data,Data,……]
    '''
    temp=pd.concat([result, mc], join='inner', axis=1).dropna()
    temp=temp.sort_values(by=temp.columns[1], ascending=False)
    cut=temp.shape[0]//ggg
    cut=list(np.array(range(ggg))*cut)
    cut.append(temp.shape[0])
    return [temp.iloc[cut[i]:cut[i+1],0] for i in range(len(cut)-1)]


def NWtest(a, lags=5):
    '''
    序列异于0的NW-t检验
    :param a: 待检验序列
    :param lags: 滞后阶数
    :return: (t,p)
    '''
    adj_a = pd.DataFrame(a)
    adj_a = adj_a.dropna()
    if len(adj_a) > 0:
        adj_a = adj_a.astype(float)
        adj_a = np.array(adj_a)
        model = sm.OLS(adj_a, [1]*len(adj_a)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        return float(model.tvalues), float(model.pvalues)
    else:
        return [np.nan]*2


def NWsummary(data_e,data_v,lags=5,hedge=False):
    '''
    对收益率或其他特征序列序列做NW-t检验
    :param data_e: 等权每期收益dataframe（未累乘）
                index: 日期   columns:分组  value:每组当期收益
    :param data_v: 市值加权每期收益dataframe（未累乘）
                index: 日期   columns:分组  value:每组当期收益
    :param lags: 滞后阶数
    :param hedge: 是否计算h-l组
    :return: DataFrame
    '''
    col = data_v.columns.tolist()
    temp_v = np.array(data_v)
    temp_e = np.array(data_e)
    nt = []
    for i in range(data_v.shape[1]):
        avg = np.nanmean(np.array(temp_v[:,i]))
        nwt = NWtest(np.array(temp_v[:,i]),lags)[0]
        avg_ = np.nanmean(np.array(temp_e[:, i]))
        nwt_ = NWtest(np.array(temp_e[:, i]), lags)[0]
        nt.append([avg,nwt,avg_,nwt_])
    if hedge:
        col.append(str(col[0])+'-'+str(col[-1]))
        avg = np.nanmean(np.array(temp_v[:, 0]-temp_v[:, -1]))
        nwt = NWtest(np.array(temp_v[:, 0]-temp_v[:, -1]), lags)[0]
        avg_ = np.nanmean(np.array(temp_e[:, 0] - temp_e[:, -1]))
        nwt_ = NWtest(np.array(temp_e[:, 0] - temp_e[:, -1]), lags)[0]
        nt.append([avg, nwt,avg_,nwt_])
    nt = pd.DataFrame(nt,columns=['市值加权均值','市值加权nw-t','等权均值','等权nw-t'],index=col)
    return nt


def get_index_comprofit(index, date):
    """
    获取一个或者多个指数的行情数据
    :param index 指数代码
    :param date list 日期列表['','',...]
    :return 返回series index为日期,value为累积收益;
    """
    temp = [float(get_index_price(index, start_date=None, end_date=date[xx], count=1)['close']) for xx in
            range(0,len(date))]
    data = [temp[xx + 1] / temp[xx] for xx in range(len(temp) - 1)]
    # data = np.nancumprod(np.array(data))
    data = pd.Series(data,index=date[0:-1])
    data = data.fillna(0)
    return data


def draw_acm(data,path,filename,reverse=False,f='9M'):
    '''
    分组累计净值曲线
    :param data: 每期收益dataframe（未累乘）
                index: 日期   columns:分组  value:每组当期收益
    :param path: 图片存放路径
    :param filename: 图片名称
    :param reverse: 是否反转（对冲改为最后组减第一组）
    :param f: 横坐标频率
    '''
    proto = np.array(data.copy().T)
    proto = np.nancumprod(proto,axis=1)
    mean = proto.mean(axis=0)
    proto = proto# - mean
    start_time =pd.to_datetime(data.index[0], format='%Y-%m-%d')
    end_time = pd.to_datetime(data.index[-1], format='%Y-%m-%d')
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文(windows)
    plt.rcParams['axes.unicode_minus'] = False# 用来正常显示负号
    fig = plt.figure(figsize=(15, 9), dpi=100)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    x = [pd.to_datetime(str(xx), format='%Y-%m-%d') for xx in data.index.values.tolist()]
    plt.xticks(pd.date_range(start_time,end_time, freq=f),rotation=45)
    color = ['b', 'g', 'grey', 'c', 'm', 'y', 'k', 'pink','violet', 'brown','w','r']
    if reverse:
        for i in range(proto.shape[0]):
            ax.plot(x, proto[len(proto)-1-i,:], color=color[i], label=i + 1)
    else:
        for i in range(proto.shape[0]):
            ax.plot(x, proto[i], color=color[i], label=i + 1)
    plt.xlabel('时间', fontsize=14)
    plt.ylabel("净值曲线", fontsize=16)
    plt.title(filename, fontsize=25, color='black',pad=20)
    plt.gcf().autofmt_xdate()
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(path,filename + '.png'))
    return proto

def draw_hedge(data,short,path,filename,reverse=False,f='9M'):
    '''
    对冲净值曲线
    :param data: 每期收益dataframe（未累乘）
                index: 日期   columns:分组  value:每组当期收益
    :param short: series index 日期 value空头收益率
    :param path: 图片存放路径
    :param filename: 图片名称
    :param reverse: 是否反转（对冲改为最后组多头）
    :param f: 横坐标频率
    '''
    proto = np.array(data.copy().T)
    short = np.array(short.T)
    # proto = np.nancumprod(proto, axis=1)
    hedge1 = proto[0] - proto[-1]
    if reverse:
        hedge2 = proto[-1]-short
        hedge1 = -hedge1
    else:
        hedge2 = proto[0]-short
    hedge1 = np.nancumprod(hedge1 + 1)-1
    hedge2 = np.nancumprod(hedge2 + 1)-1
    start_time = pd.to_datetime(data.index[0], format='%Y-%m-%d')
    end_time = pd.to_datetime(data.index[-1], format='%Y-%m-%d')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文(windows)
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(figsize=(15, 9), dpi=100)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    x = [pd.to_datetime(str(xx), format='%Y-%m-%d') for xx in data.index.values.tolist()]
    plt.xticks(pd.date_range(start_time, end_time, freq=f), rotation=45)
    color = [ 'r', 'purple']
    ax.plot(x, hedge1, color=color[0], label='组合对冲')
    ax.plot(x, hedge2, color=color[1], label='指数对冲')
    plt.xlabel('时间', fontsize=14)
    plt.ylabel("净值曲线", fontsize=16)
    plt.title(filename, fontsize=25, color='black', pad=20)
    plt.gcf().autofmt_xdate()
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(path, filename + '.png'))


def draw_bar(data_e,data_v,path,filename):
    '''
    分组收益bar
    :param data_e: 等权每期收益dataframe（未累乘）
                index: 日期   columns:分组  value:每组当期收益
    :param data_v: 市值加权每期收益dataframe（未累乘）
                index: 日期   columns:分组  value:每组当期收益
    :param path: 图片存放路径
    :param filename: 图片名称
    '''
    ew_r = np.array(data_e.copy().T)  # 每期收益
    vw_r = np.array(data_v.copy().T)
    g = ew_r.shape[0] #分组数
    name_list = [str(xx) for xx in range(1,1+g)]
    ew = np.nanmean(ew_r, axis=1)-1  # 平均收益
    vw = np.nanmean(vw_r, axis=1)-1
    x = list(range(g))
    total_width, n = 0.8, 2
    width = total_width / n
    fig = plt.figure(figsize=(15, 9), dpi=100)
    plt.bar(x, ew, width=width, label='ew', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, vw, width=width, label='vw', tick_label=name_list, fc='r')
    plt.legend()
    plt.xlabel('按照'+filename+'划分的'+str(g)+'个投资组合')
    plt.ylabel("月均收益率")
    plt.title(filename,color='black')
    plt.savefig(os.path.join(path,filename + '.png'))


def feature_summary(factor,mc,roa,fname):
    '''
    特征统计
    :param factor: 因子--dict，key为'%Y-m-%d'，value为list[组0,组1,……]
    :param mc: 市值——dict，key为'%Y-m-%d'，value为list[组0,组1,……]
    :param roa: roa——dict，key为'%Y-m-%d'，value为list[组0,组1,……]
    :param fname: 因子名
    :return: DataFrame
    '''
    factor = np.nanmean(pd.DataFrame(factor), axis=1)
    mc_ = np.nanmean(pd.DataFrame(mc), axis=1)
    roa_= np.nanmean(pd.DataFrame(roa), axis=1)
    return pd.DataFrame([factor,mc_,roa_],index=[fname,'总市值','ROA'])

def Close_price(factor):
    """
    factor: dataframe
            index 日期
            columns 股票代码
    return： dataframe
            index 日期
            columns 股票代码
            value 收盘价
    """
    stocks = factor.columns.tolist()
    timelist = factor.index.tolist()
    datelist = []
    for time in timelist:
        tmp = parse(time)
        datelist.append(tmp)
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_daily_price"]
    myquery = ({"code": {"$in": tuple(stocks)}, "date": {"$in": tuple(datelist)}})
    mydoc = mycol.find(myquery)
    price_data = pd.DataFrame(list(mydoc))
    price_data['date'] = price_data['date'].apply(lambda x: datetime.strftime(x,'%Y-%m-%d'))
    factor_price = pd.pivot_table(price_data, index="date", columns="code", values="close", dropna=False)
    return factor_price


def Market_value(factor,circulating_market=False):
    """
    factor: dataframe
            index 日期
            columns 股票代码
    return： dataframe
            index 日期
            columns 股票代码
            value 市值
    """
    if circulating_market:
        market = "circulating_market_cap"
    else:
        market = "market_cap"
    stocks = factor.columns.tolist()
    timelist = factor.index.tolist()
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_daily_valuation"]
    myquery = ({"code": {"$in": tuple(stocks)}, "date": {"$in": tuple(timelist)}})
    mydoc = mycol.find(myquery)
    valuation_data = pd.DataFrame(list(mydoc))
    factor_market = pd.pivot_table(valuation_data, index="date", columns="code", values=market, dropna=False)
    return factor_market


def Group_by(factor,industry_type='sw_l1'):
    """
    factor: dataframe
            index 日期
            columns 股票代码
    return： dataframe
            index 日期
            columns 股票代码
            value 行业
    """
    stocks = factor.columns.tolist()
    timelist = factor.index.tolist()
    timelist = map(parse,timelist)
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_daily_industry"]
    myquery = ({"code": {"$in": tuple(stocks)}, "date": {"$in": tuple(timelist)}})
    mydoc = mycol.find(myquery)
    stock_industry = pd.DataFrame(list(mydoc))
    stock_industry = stock_industry.pivot(index="date", columns="code", values=industry_type)
    return stock_industry


def bi_test(data,x_num=4,y_num=3,x_ascend=True,y_ascend = True,market=False):
    '''
    单周期双排
    :param data: <DataFrame>,index=scurity,columns=factor1,factor2,profit,market
    :param market: True or False 市值加权
    :param x_num: 横向分组
    :param y_num: 纵向分组
    :param ascend: 是否升序
    :return: <dict>
    '''
    data2 = pd.DataFrame(index=data.index,columns=['fac'])
    x_group = data.columns[0]
    y_group = data.columns[1]
    pro = data.columns[2]
    mar = data.columns[3]
    num = data.shape[0]//x_num
    cut = list(np.array((range(x_num))) * num)
    cut.append(data.shape[0])
    data = data.sort_values(by=y_group, ascending=y_ascend)
    ret = {}
    for j in range(y_num):
        ret[j] = {}
    for i in range(x_num):
        data1 = data.iloc[cut[i]:cut[i+1],:]
        num1 = data1.shape[0] // y_num
        cut1 = list(np.array((range(y_num))) * num1)
        cut1.append(data1.shape[0])
        data1 = data1.sort_values(by=x_group, ascending=x_ascend)
        for j in range(y_num):
            data2.loc[data1.iloc[cut1[j]:cut1[j + 1]].index.tolist(),:] = i*x_num+j
            if market is False:
                ret[j][i] = np.nanmean(data1.iloc[cut1[j]:cut1[j + 1]][pro])
            else:
                ret[j][i] = np.nansum(
                    data1.iloc[cut1[j]:cut1[j + 1]][pro] * data1.iloc[cut1[j]:cut1[j + 1]][mar]) / np.nansum(
                    data1.iloc[cut1[j]:cut1[j + 1]][mar])
    return ret,data2


def bi_summary(bi,name1,name2):
    '''
    多周期双排整理
    :param bi: <dict>,{0:bi_test返回值,1:bi_test返回值,……}
    :param y_factor: <str/numeric>纵向划分的因子名称
    :param x_factor: <str/numeric>横向划分的因子名称
    :return:
    '''
    temp = [profit_2df(bi[t])-1 for t in range(len(bi))]
    ans = sum(temp)/len(temp)
    group_y = ans.shape[0]
    group_x = ans.shape[1]
    ans.loc['0-'+str(group_y-1)] = list(ans.iloc[0,:]-ans.iloc[-1,:])
    ans['0-' + str(group_x-1)] = list(ans.iloc[:, 0] - ans.iloc[:, -1])
    ans.iloc[-1,-1] = np.nan
    ans.index = [name1+str(x) for x in ans.index]
    ans.columns = [name2 + str(x) for x in ans.columns]
    newind = [ind for ind in ans.index]+['tvalue']
    newcol = [col for col in ans.columns] + ['tvalue']
    ans = ans.reindex(index=newind,columns=newcol)
    for i in range(group_x):
        ts_l = np.array([xx.iloc[0,i] for xx in temp])
        ts_s = np.array([xx.iloc[-1, i] for xx in temp])
        ts = ts_l-ts_s
        ans.iloc[group_y+1, i] = stats.ttest_1samp(ts, 0)[0]
    for i in range(group_y):
        ts_l = np.array([xx.iloc[i, 0] for xx in temp])
        ts_s = np.array([xx.iloc[i, -1] for xx in temp])
        ts = ts_l - ts_s
        ans.iloc[i, group_x+1] = stats.ttest_1samp(ts,0)[0]
    return ans

