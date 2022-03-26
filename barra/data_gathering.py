from readdb import *
from util import *
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_data(start,end,index = "000905.XSHG",store = True):
    # index = "000905.XSHG"
    # start_date = '2012-06-01'
    # end_date = '2016-6-01'
    tl_d = get_trading_day_list(start_time=start, end_time=end, frequency='day')
    all_stocks = get_periods_index_stocks(index, start_date=start, end_date=end)

    factor = pd.DataFrame(index=tl_d, columns=all_stocks)

    barra = get_barra(all_stocks, start_date=start, end_date=end)
    # stock_industry = get_stock_industry(stocks=all_stocks, industry_type='sw_l1', date=start)
    # circulating_market select stock whet
    factor_market = Market_value(factor, circulating_market=True)
    factor_prices = Close_price(factor)
    # -------- stock return
    ret = factor_prices.pct_change(fill_method=None).shift(-1)

    style_nums = barra.shape[0]-2
    print('==============capital processing===============')
    def foo_captial(name, x):
        try:
            x['captial'] = factor_market.loc[:, name].dropna(axis=0).values
        except:
            #         print(name)
            #         print(len(x),len(factor_market.loc[:,name].dropna(axis=0).values))
            #         x['captial'] = 'Nan'
            captial = factor_market.loc[:, name].dropna(axis=0).values
            captial = np.concatenate((captial, np.array(captial[-1]).reshape(-1)))
            x['captial'] = captial
        return x

    barra = barra.groupby('code').apply(lambda x: foo_captial(x.name, x))


    print('==============return processing===============')
    def foo_ret(name, x):
        '''
        ret: stock return
        '''
        df = pd.DataFrame({'date': pd.to_datetime(ret.index), 'ret': ret[name].values})
        final_df = df.merge(x, how='inner', on='date')
        return final_df

    barra = barra.groupby('code').apply(lambda x: foo_ret(x.name, x))
    barra = barra.reset_index(drop=True)

    # all_stocks = get_periods_index_stocks(index, start_date=start_date, end_date=end_date)
    stock_industry_all = None
    print('==' * 10 + 'get the industry factor' + '==' * 10)
    for date in tqdm(tl_d):
        #     all_stocks = get_periods_index_stocks(index, start_date=date)
        stock_industry = get_stock_industry(stocks=all_stocks, industry_type='sw_l1', date=date)
        #     stock_industry['date'] = date
        stock_industry = pd.DataFrame({'code': stock_industry.index, 'industry': stock_industry.values})
        stock_industry['date'] = date
        stock_industry_all = pd.concat((stock_industry_all, stock_industry), axis=0)
    stock_industry_all.columns = ['code', 'industry_names', 'date']
    stock_industry_all['date'] = pd.to_datetime(stock_industry_all['date'] )


    def foo_industry(name, x):
        industry_info = stock_industry_all[stock_industry_all['date'] == name][['code', 'industry_names']]
        return x.merge(industry_info, how='inner', on='code')

    barra = barra.groupby('date').apply(lambda x: foo_industry(x.name, x))
    barra = barra.reset_index(drop=True)
    indus = pd.get_dummies(barra.industry_names)
    indus_name_list = list(indus.columns)
    barra = pd.concat((barra, indus), axis=1).drop('industry_names', axis=1)
    df = barra[['date', 'code', 'ret', 'captial',]+indus_name_list+['size','beta','momentum','residual_volatility','non_linear_size',
                                                                    'book_to_price_ratio','liquidity','earnings_yield','growth','leverage']]

    print(f'{len(indus_name_list)} industry factors , 10 style factors')
    if store:
        df.to_csv('barra_data.csv',index = None)
    return df,len(indus_name_list),style_nums