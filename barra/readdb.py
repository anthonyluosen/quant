from datetime import datetime
import pandas as pd
import numpy as np
import pymongo
from dateutil.parser import *
from threading import Thread

def get_trade_days(start_date=None, end_date=None, count=None):
    """
    获取指定日期范围内的所有交易日
    start:str or datetime,起始时间，与count二选一
    end:str or datetime，终止时间
    :return numpy.ndarray, 包含指定的 start_date 和 end_date, 默认返回至 datatime.date.today() 的所有交易日
    """
    if start_date and count:
        raise NameError("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_index_daily_price"]
    if count:
        myquery = ({"code": '000300.XSHG', "date": {"$lte": parse(end_date)}})
        mydoc = mycol.find(myquery).sort("date", -1).limit(count)
    else:
        myquery = ({"code": '000300.XSHG', "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
        mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc))
    data = data["date"].apply(lambda x: x.strftime('%Y-%m-%d')).values
    data.sort()
    return data


def get_price(security, start_date=None, end_date=None, count=None ):
    """
    获取一支或者多只证券的行情数据
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 返回pandas.DataFrame对象
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_daily_price"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_date_price(security, date=None ):
    """
    获取一支或者多只证券的行情数据
    :param security 一支证券代码或者一个证券代码的list
    :return 返回pandas.DataFrame对象
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    date = map(parse, date)
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_daily_price"]
    myquery = ({"code": {"$in": tuple(security)}, "date": {"$in": tuple(date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_min_price(security, start_date=None, end_date=None, count=None):
    """
    获取一支或者多只证券的分钟行情数据
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 返回pandas.DataFrame对象
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_min_price"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_min_price1(security,start_date=None, end_date=None):
    """
    获取一支或者多只证券的分钟行情数据
    :param security 一支证券代码或者一个证券代码的list
    :return 返回pandas.DataFrame对象
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["minprice"]
    data = pd.DataFrame()
    for code in security:
        mycol = mydb[code]
        try:
            myquery = ({"date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
            mydoc = mycol.find(myquery)
            temp = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
        except:
            temp = pd.DataFrame()
        data = data.append(temp)
    return data


def get_index_price(security, start_date=None, end_date=None, count=None ):
    """
    获取一个或者多个指数的行情数据
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 返回pandas.DataFrame对象;
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_index_daily_price"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_index_min_price(security, start_date=None, end_date=None, count=None ):
    """
    获取一支或者多只指数的分钟行情数据
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 返回pandas.DataFrame对象;
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_index_min_price"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_barra(security, start_date=None, end_date=None, count=None):
    """
    获取一支或者多只证券的barra风格因子
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 则返回pandas.DataFrame对象;
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_barra_factor"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_valuation(security, start_date=None, end_date=None, count=None):
    """
    获取一支或者多只证券的估值
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 返回pandas.DataFrame对象;
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_daily_valuation"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": start_date, "$lte": end_date}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_daily_quarterly_fundamental(security, start_date=None, end_date=None, count=None,keep=False):
    """
    获取一支或者多只证券的季报
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :param keep(同一天更新四季度和一季度报) False两者都保留，'first'只保留四季报，'last'只保留一季度报
    :return 返回pandas.DataFrame对象;
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["daily quarterly fundamental"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": start_date, "$lte": end_date}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    if keep:
        data = data.drop_duplicates(subset=['code', 'date'], keep=keep)
    return data


def get_daily_yearly_fundamental(security, start_date=None, end_date=None, count=None):
    """
    获取一支或者多只证券的年报
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 返回pandas.DataFrame对象;
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["daily yearly fundamental"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": start_date, "$lte": end_date}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_index(security, start_date=None, end_date=None):
    """
    获取指数成份股
    :param security 一支证券代码或者一个证券代码的list
    :param start_date 开始时间
    :param end_date  结束时间
    :return 返回pandas.DataFrame对象
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if end_date is None:
        end_date = str(datetime.today())
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_month_index"]
    myquery = ({"index_code": {"$in": security}, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_fundamental(security, statDate=None,date=None,keep='last'):
    """
    获取财务数据
    :param security 一支证券代码或者一个证券代码的list
    :param date      和statDate参数只能传入一个: 查询指定日期date收盘后所能看到的最近一个季度的数据
    :param statDate  获取季报或年报财务数据
                     季度: 格式是: 年 +  季度最后一天, 例如: '2015-03-31', '2013-06-30'.
                     年份: 格式就是年份的数字, 例如: '2015', '2016'.
    :return 返回pandas.DataFrame对象
    """
    if statDate and date:
        raise NameError("statDate 参数与 date 参数只能二选一")
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    if statDate is None:
        mycol = mydb["stock_quarterly_fundamental"]
        year = int(date[:4])
        startdate = str(year - 1) + date[4:]
        myquery = ({"code": {"$in": security}, "pubDate":  {"$gt": startdate,"$lt": date}})
        mydoc = mycol.find(myquery)
        data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
        if keep:
            data = data.drop_duplicates(subset=['code', 'date'], keep=keep)
        return data
    else:
        if len(statDate)== 10:
            mycol = mydb["stock_quarterly_fundamental"]
            myquery = ({"code": {"$in": security}, "statDate": statDate})
        if len(statDate)== 4:
            mycol = mydb["stock_yearly_fundamental"]
            myquery = ({"code": {"$in": security}, "statDate": statDate+'-12-31'})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    return data


def get_period_fundamental(security, start_date, end_date,keep='last'):
    '''
    获取一段期间的所有财务数据，以['code','pubDate']为键，重复时选最新会计期
    :param security: [code,code,......]
    :param start_date: 'YYYY-mm-dd'
    :param end_date: 'YYYY-mm-dd'
    :return: <DataFrame>一段期间的所有财务数据，以['code','pubDate']为键，重复时选最新会计期
    '''
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_quarterly_fundamental"]
    myquery = ({"code": {"$in": security}, "pubDate":  {"$gt": start_date,"$lt": end_date}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    if keep:
        data = data.drop_duplicates(subset=['code', 'pubDate'], keep=keep)
    return data


def get_year_fundamental(security, start_date, end_date):
    '''
    获取一段期间的所有财务数据，以['code','pubDate']为键，重复时选最新会计期
    :param security: [code,code,......]
    :param start_date: 'YYYY-mm-dd'
    :param end_date: 'YYYY-mm-dd'
    :return: <DataFrame>一段期间的所有财务数据，以['code','pubDate']为键，重复时选最新会计期
    '''
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_yearly_fundamental"]
    myquery = ({"code": {"$in": security}, "pubDate":  {"$gt": start_date,"$lt": end_date}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    data = data.drop_duplicates(subset=['code','pubDate'], keep='last')
    return data


def get_call_auction(security, start_date=None, end_date=None, count=None):
    """
    获取一支或者多只证券集合竞价
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 返回pandas.DataFrame对象;
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_call_auction"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    data.date = data.date.map(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    return data


def get_money_flow(security, start_date=None, end_date=None, count=None):
    """
    获取一支或者多只证券资金流
    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间
    :return 返回pandas.DataFrame对象;
    """
    if isinstance(security, str):
        security = (security,) * 2
    else:
        security = tuple(security)
    if count and start_date:
        raise NameError("(start_date, count) only one param is required")
    if not (count is None or count > 0):
        raise NameError("count 参数需要大于 0 或者为 None")
    if end_date is None:
        end_date = str(datetime.today())
    if count:
        start_date = get_trade_days(end_date=end_date, count=count)[0]
    myclient = pymongo.MongoClient("mongodb://192.168.31.50:27017/")
    mydb = myclient["jialiang"]
    mycol = mydb["stock_money_flow"]
    myquery = ({"code": {"$in": security}, "date": {"$gte": parse(start_date), "$lte": parse(end_date)}})
    mydoc = mycol.find(myquery)
    data = pd.DataFrame(list(mydoc)).drop(['_id'], axis=1)
    data.date = data.date.map(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    return data