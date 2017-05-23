# 多因子选股模型
# 先导入所需要的程序包
# import tushare as ts
import statsmodels.api as sm
from statsmodels import regression
import datetime
import numpy as np
import pandas as pd
import time
from jqdata import *
from pandas import Series, DataFrame

'''
================================================================================
总体回测前
================================================================================
'''
# 本程序实现了东方证券机器学习因子库中的  {“乒乓球反转因子” PPReversal, CGO Capital Gains Overhang
                                        #   TO 流通股本日均换手率, EPS_GR(EPS Growth Ratio)每股收益增长率（年度同比）
                                        #   Revenue Growth Ratio营业收入增长率 季度同比 , 净利润增长率季度同比 net_profit Growth Ratio
                                        #   非流动性因子illiquidity factor(ILLIQ) , Fama-French 三因子、五因子模型、
                                        #   市值因子}


#总体回测前要做的事情
def initialize(context):
    set_params()    #1 设置策参数
    set_variables() #2 设置中间变量
    set_backtest()  #3 设置回测条件

#1 设置策略参数
def set_params():
    # 单因子测试时g.factor不应为空
    g.factor = 'AllFactor'          # 当前回测的单因子
    g.shift = 21                    # 设置一个观测天数（天数）
    g.stock_num = 50.0              # 设置买卖股票数目，行业中性会导致股票数目偏多一些,坑爹的python2的除法，一定要设置成浮点数。
    # 设定选取sort_rank： True 为最大，False 为最小
    g.sort_rank = True
    # 多因子合并称DataFrame，单因子测试时可以把无用部分删除提升回测速度
    # 定义因子以及排序方式，默认False方式为降序排列，原值越大sort_rank排序越小
    g.factors = {'AllFactor': False }
    # 计算因子均值和相关系数的调仓周期数
    g.periods=6
    # 选择因子加权方法
    g.linear_regression=False    # 线性回归
    g.optimal_choose=True         # 最优化
    # 是否行业中性化处理
    g.Industry_Neutral=True
    # 是否去掉创业板True则去掉
    g.No_GEM=True
    # 是否去掉上市不满三个月的新股True则去掉
    g.No_new_stock=True
    # 参与回测的因子名称，不添加因子名称则不参与回测，MTM1同时也是一个月涨跌幅，不能去掉
    g.list_factor_name=['BP','MTM1','MTM3','PPReversal','CGO','TO','EPS_GR','RGR',
                        'NPGR','ILLIQ','FF','CMC']
    # 设置全局变量用于保存计算结果
    # 定义全局变量 保存多因子信息
    # 去极值、标准化后的因子值
    g.panel_factor=pd.Panel()
    # 去极值、未标准化的因子值
    g.panel_factor_value=pd.Panel()
    # 将全局变量g.list_factor_name换个名字
    series_global_temp=Series(g.list_factor_name)
    series_global_temp='series_'+series_global_temp
    g.list_index_name=list(series_global_temp)
    # 定义全局变量存储IC值
    g.df_IC=DataFrame(index=g.list_factor_name)
    # 计数全局变量
    g.count=0


#2 设置中间变量
def set_variables():
    g.feasible_stocks = []  # 当前可交易股票池
    g.if_trade = False      # 当天是否交易
    g.num_stocks = 50        # 设置持仓股票数目

    
#3 设置回测条件
def set_backtest():
    set_benchmark('399300.XSHE')       # 设置为基准 沪深300
    set_option('use_real_price', True) # 用真实价格交易
    log.set_level('order', 'error')    # 设置报错等级

'''
================================================================================
每天开盘前
================================================================================
'''
#每天开盘前要做的事情
def before_trading_start(context):
    # 获得当前日期
    day = context.current_dt.day
    # 获得前一个交易日日期
    yesterday = context.previous_date
    # 调仓日期为上一个交易日后的一个交易日
    rebalance_day = shift_trading_day(yesterday, 1)
    # 如果上一个交易日与下一个交易日月份不同开始交易
    if yesterday.month != rebalance_day.month:
        # 获取调整为day == rebalance_day.day更好？
        if yesterday.day > rebalance_day.day:
            g.if_trade = True 
            #5 设置可行股票池：获得当前开盘的股票池并剔除当前或者计算样本期间停牌的股票
            list_stocks = list(get_all_securities(['stock']).index)
            # 去掉创业板
            if g.No_GEM:
                list_stocks = filter_gem_stock(list_stocks)
            # 去掉上市不满三个月的新股
            if g.No_new_stock:
                list_stocks = filter_new_stock(context, list_stocks)
            g.feasible_stocks = set_feasible_stocks(list_stocks, g.shift,context)
    		#6 设置手续费与手续费
            set_slip_fee(context)
            # 购买股票为可行股票池对应比例股票
            # g.num_stocks = int(len(g.feasible_stocks)*g.precent)

#剔除上市不满3个月的股票
def filter_new_stock(context, stock_list):
    tmpList = []
    for stock in stock_list :
        days_public=(context.current_dt.date() - get_security_info(stock).start_date).days
        # 上市未超过三个月
        if days_public > 93:
            tmpList.append(stock)
    return tmpList

# 过滤掉创业板
def filter_gem_stock(stock_list):
    return [stock for stock in stock_list if stock[0:3] != '300']

#4
# 某一日的前shift个交易日日期 
# 输入：date为datetime.date对象(是一个date，而不是datetime)；shift为int类型
# 输出：datetime.date对象(是一个date，而不是datetime)
def shift_trading_day(date,shift):
    # 获取所有的交易日，返回一个包含所有交易日的 list,元素值为 datetime.date 类型.
    tradingday = get_all_trade_days()
    # 得到date之后shift天那一天在列表中的行标号 返回一个数
    shiftday_index = list(tradingday).index(date)+shift
    # 根据行号返回该日日期 为datetime.date类型
    return tradingday[shiftday_index]

#5    
# 设置可行股票池
# 过滤掉当日停牌的股票,且筛选出前days天未停牌股票
# 输入：stock_list为list类型,样本天数days为int类型，context（见API）
# 输出：list=g.feasible_stocks
def set_feasible_stocks(stock_list,days,context):
    # 得到是否停牌信息的dataframe，停牌的1，未停牌得0
    suspened_info_df = get_price(list(stock_list), 
                       start_date=context.current_dt, 
                       end_date=context.current_dt, 
                       frequency='daily', 
                       fields='paused'
    )['paused'].T
    # 过滤停牌股票 返回dataframe
    unsuspened_index = suspened_info_df.iloc[:,0]<1
    # 得到当日未停牌股票的代码list:
    unsuspened_stocks = suspened_info_df[unsuspened_index].index
    current_data = get_current_data()
    return [stock for stock in unsuspened_stocks if  not 
        current_data[stock].is_st and 
        'ST' not in current_data[stock].name and 
        '*' not in current_data[stock].name and 
        '退' not in current_data[stock].name]


    
#6 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0)) 
    # 根据不同的时间段设置手续费
    dt=context.current_dt
    
    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, 
                                sell_cost=0.0013, 
                                min_cost=5)) 
        
    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, 
                                sell_cost=0.002, 
                                min_cost=5))
            
    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, 
                                sell_cost=0.003, 
                                min_cost=5))
                
    else:
        set_commission(PerTrade(buy_cost=0.003, 
                                sell_cost=0.004, 
                                min_cost=5))
'''
================================================================================
每天交易时
================================================================================
'''
def handle_data(context,data):
	# 如果为交易日
    if g.if_trade == True: 
	    #8 获得买入卖出信号，输入context，输出股票列表list
	    # 字典中对应默认值为false holding_list筛选为true，则选出因子得分最大的
        holding_list = get_stocks(g.feasible_stocks, 
                                context, 
                                g.factors, 
                                asc = g.sort_rank, 
                                factor_name = g.factor)
                                
        # print len(holding_list)
        #9 重新调整仓位，输入context,使用信号结果holding_list
        if g.count>g.periods:
            rebalance(context, holding_list)
	g.if_trade = False

#7 获得因子信息
# stocks_list调用g.feasible_stocks factors调用字典g.factors
# 输出所有对应数据和对应排名，DataFrame
def get_factors(stocks_list, context, factors):
    # 从可行股票池中生成股票代码列表
    df_all_raw = pd.DataFrame(stocks_list)
    # 修改index为股票代码
    df_all_raw['code'] = df_all_raw[0]
    df_all_raw.index = df_all_raw['code']
    # 格式调整，没有一步到位中间有些东西还在摸索，简洁和效率的一个权衡
    del df_all_raw[0]
    stocks_list300 = list(df_all_raw.index)
    # 每一个指标量都合并到一个dataframe里
    for key,value in g.factors.items():
        # 构建一个新的字符串，名字叫做 'get_df_'+ 'key'
        tmp='get_df' + '_' + key
        # 声明字符串是个方程
        aa = globals()[tmp](stocks_list, context, value)
        # 合并处理
        df_all_raw = pd.concat([df_all_raw,aa], axis=1)
    # 删除code列
    del df_all_raw['code']
    # 对于新生成的股票代码取list
    stocks_list_more = list(df_all_raw.index)
    # 可能在计算过程中并如的股票剔除
    for stock in stocks_list_more[:]:
        if stock not in stocks_list300:
            df_all_raw.drop(stock)
    return df_all_raw

# 8获得调仓信号
# 原始数据重提取因子打分排名
def get_stocks(stocks_list, context, factors, asc, factor_name):
    # 7获取原始数据
    df_all_raw1 = get_factors(stocks_list, context, factors)
    # 根据factor生成列名
    score = factor_name + '_' + 'sorted_rank'
    stocks = list(df_all_raw1.sort(score, ascending = asc).index)
    return stocks

# 9交易调仓
# 依本策略的买入信号，得到应该买的股票列表
# 借用买入信号结果，不需额外输入
# 输入：context（见API）
def rebalance(context, holding_list):
    # 每只股票购买金额
    every_stock = context.portfolio.portfolio_value/g.num_stocks
    # 空仓只有买入操作
    if len(list(context.portfolio.positions.keys()))==0:
        # 原设定重scort始于回报率相关打分计算，回报率是升序排列
        for stock_to_buy in list(holding_list)[0:g.num_stocks]: 
            order_target_value(stock_to_buy, every_stock)
    else :
        # 不是空仓先卖出持有但是不在购买名单中的股票
        for stock_to_sell in list(context.portfolio.positions.keys()):
            if stock_to_sell not in list(holding_list)[0:g.num_stocks]:
                order_target_value(stock_to_sell, 0)
        # 因order函数调整为顺序调整，为防止先行调仓股票由于后行调仓股票占金额过大不能一次调整到位，这里运行两次以解决这个问题
        for stock_to_buy in list(holding_list)[0:g.num_stocks]: 
            order_target_value(stock_to_buy, every_stock)
        for stock_to_buy in list(holding_list)[0:g.num_stocks]: 
            order_target_value(stock_to_buy, every_stock)
            

#行业中性化处理，按照证监会标准分为十八个行业，选择股票时从相应行业选取相应比例的股票，
#如果某一行业股票过少不足一支时取一支,输入一个series结构index为股票代码，内容为选股因子,再输入因子名称格式为字符串
def INDUSTRY_SORTED(stock_list,df_FACTOR,FACTOR_NAME):
    #获得行业信息
    # 将stock_list的股票代码转换为国泰安的格式
    def stock_code_trun(stock_list):
        gta_stock_list=[]
        # stock_list=list(stock_list)
        for stock in stock_list:
            temp_stock=stock[:6]
            gta_stock_list.append(temp_stock)
        return gta_stock_list
    gta_stock_list = stock_code_trun(stock_list)    
    #获取行业信息,取上市所有股票，深市为SZSE，一次最多返回3000条信息，因此要筛掉无用信息 
    df_stk = gta.run_query(query(gta.STK_INSTITUTIONINFO.SYMBOL , gta.STK_INSTITUTIONINFO.INDUSTRYCODE
                            ).filter(gta.STK_INSTITUTIONINFO.SYMBOL.in_(gta_stock_list)))
    #得到一个series存放股票所在行业
    series_industry=df_stk.ix[:,'INDUSTRYCODE']
    #将股票代码与行业信息对应
    #将股票代码转化为聚宽的格式
    list_code_temp=list()
    for symbol_temp in df_stk.ix[:,'SYMBOL']:
      list_code_temp.append(normalize_code(symbol_temp))
    series_industry.index=list_code_temp
    #去掉不需要的行业信息
    temp=list(df_FACTOR.index)
    series_industry=series_industry[temp]
    #丢掉没有行业信息的数据
    series_industry=series_industry.dropna()
    #将相应的缺失行业的因子信息一并删掉
    temp=list(series_industry.index)
    df_FACTOR=df_FACTOR.ix[temp]
    #去掉行业编码只留下行业类别，例如A01去掉01留下A
    list_INDUSTRY_temp=list()
    for industrycode_temp in series_industry:
      list_INDUSTRY_temp.append(industrycode_temp[0])
    #将股票代码、factor、行业信息写入一个DataFrame
    df_temp_p=DataFrame(columns=[FACTOR_NAME,'INDUSTRY'])
    df_temp_p[FACTOR_NAME]=df_FACTOR[FACTOR_NAME]
    df_temp_p['INDUSTRY']=list_INDUSTRY_temp
    # 删除空值
    df_temp_p=df_temp_p.dropna()
    #行业中性处理，防止股票集中在个别行业，取每个行业的股票不超过总股票数的g.precent支
    #列出所有行业分类
    list_INDUSTRY=list('ABCDEFGHIJKLMNPQRS')
    #筛选行业
    df_INDUSTRY_selected=DataFrame(columns=[FACTOR_NAME,'INDUSTRY'])
    #本程序里排序rank和sort不一样rank=True时sort=False排序才一样
    g.precent = g.stock_num / len(df_temp_p)
    sort_asc=not g.sort_rank
    for industry in list_INDUSTRY:
        df_temp=df_temp_p[df_temp_p.INDUSTRY==industry]
        if len(df_temp)>0:
            df_temp=df_temp.sort(columns=FACTOR_NAME,ascending=sort_asc)
            #选取该行业的g.percent的股票，其余股票删掉
            num_temp=len(df_temp)*g.precent
            #为避免选出的股票数少于g.num_stocks,取整再+1
            num_temp=int(num_temp)+1
            df_temp=df_temp[:num_temp]
            #添加到股票池
            df_INDUSTRY_selected=pd.concat([df_INDUSTRY_selected,df_temp])
        else:
            pass
    #有些公司可能会分到不同的行业，需要去重
    df_INDUSTRY_selected['temp_index']=df_INDUSTRY_selected.index
    df_INDUSTRY_selected=df_INDUSTRY_selected.drop_duplicates(subset='temp_index')
    #删除多余信息
    del df_INDUSTRY_selected['INDUSTRY']
    del df_INDUSTRY_selected['temp_index']
    # 全局变量赋值
    g.num_stocks = len(df_INDUSTRY_selected)
    # print '行业中性输出', df_INDUSTRY_selected
    return df_INDUSTRY_selected

# 1 账面市值比
def get_df_BP(stock_list, context, asc):
    df_BP = get_fundamentals(query(valuation.code, valuation.pb_ratio
                     ).filter(valuation.code.in_(stock_list)))
    # 获得pb倒数
    df_BP['BP'] = df_BP['pb_ratio'].apply(lambda x: 1/x)
    df_BP.index=df_BP['code']
    # 删除nan
    df_BP = df_BP.dropna()
    df_BP['BP_sorted_rank'] = df_BP['BP'].rank(ascending = asc, method = 'dense')
    return df_BP


# 2 一个月动能，输入stock_list, context, asc = True/False
# 输出：dataframe，index为code
def get_df_MTM1(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 一个shift前的交易日日期
    days_1shift_before = shift_trading_day(yest, shift = -21)
    # 获得上个交易日收盘价
    df_price_info = get_price(list(stock_list), 
                   start_date=yest, 
                   end_date=yest, 
                   frequency='daily', 
                   fields='close')['close'].T
    # 1个月前收盘价信息
    df_price_info_1shift = get_price(list(stock_list), 
                       start_date=days_1shift_before, 
                       end_date=days_1shift_before, 
                       frequency='daily', 
                       fields='close')['close'].T
    # 1月的收益率,Series
    Series_mtm1 = (df_price_info.ix[:,yest] 
                   - df_price_info_1shift.ix[:,days_1shift_before]
                   )/df_price_info_1shift.ix[:,days_1shift_before]
    #生成DataFrame
    df_MTM1=DataFrame()
    df_MTM1['MTM1']= Series_mtm1
    # 排序给出排序打分，MTM1
    df_MTM1['MTM1_sorted_rank'] = df_MTM1['MTM1'].rank(ascending = asc, method = 'dense')
    return df_MTM1
    
# 3 三个月动能，输入stock_list, context, asc = True/False
# 输出：dataframe，index为code
def get_df_MTM3(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 3个shift前的交易日日期
    days_3shift_before = shift_trading_day(yest, shift = -63)
    # 获得上个交易日收盘价
    df_price_info = get_price(list(stock_list), 
                   start_date=yest, 
                   end_date=yest, 
                   frequency='daily', 
                   fields='close')['close'].T
    # 3个月前收盘价信息
    df_price_info_3shift = get_price(list(stock_list), 
                       start_date=days_3shift_before, 
                       end_date=days_3shift_before, 
                       frequency='daily', 
                       fields='close')['close'].T
    # 3个月的收益率,Series
    Series_mtm3 = (df_price_info.ix[:,yest] 
                   - df_price_info_3shift.ix[:,days_3shift_before]
                   )/df_price_info_3shift.ix[:,days_3shift_before]
    #生成DataFrame
    df_MTM3=DataFrame()
    df_MTM3['MTM3']=Series_mtm3
    # 排序给出排序打分，MTM3
    df_MTM3['MTM3_sorted_rank'] = df_MTM3['MTM3'].rank(ascending = asc, method = 'dense')
    return df_MTM3

    
# 4 PPReversal 东方证券“乒乓球反转因子” PPReversal=5日均价/60日成交均价
# 一个月动能，输入stock_list, context, asc = True/False
# 输出：dataframe，index为code
def get_df_PPReversal(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 交易日日期
    date_5days_before = shift_trading_day(yest, shift = -5)
    date_60days_before = shift_trading_day(yest, shift = -60)
    # 获得5日均价
    df_price_5 = get_price(list(stock_list), 
                   start_date=date_5days_before, 
                   end_date=yest, 
                   frequency='daily', 
                   fields='avg').mean()
    # 60日均价
    df_price_60 = get_price(list(stock_list), 
                       start_date=date_60days_before, 
                       end_date=yest, 
                       frequency='daily', 
                       fields='avg').mean()
    # PPReversal
    df_PPReversal=df_price_5/df_price_60
    df_PPReversal.columns=['PPReversal']
    df_PPReversal=df_PPReversal.dropna()
    # # 生生series 
    # series_PPReversal= df_PPReversal['PPReversal']
    # # 生成DataFrame 
    # df_PPReversal=DataFrame()
    # # 行业中性化
    # df_PPReversal=INDUSTRY_SORTED(series_PPReversal,'PPReversal')
    # 排序给出排序打分，MTM1
    df_PPReversal['PPReversal_sorted_rank'] = df_PPReversal['PPReversal'].rank(ascending = asc, method = 'dense')
    return df_PPReversal
    
# 5 CGO Capital Gains Overhang 
# Rt+l=(Vt)(Pt) + (1-Vt)(Rt) # R考虑过去一段时间的换手率加权的平均价格 成交量越高的成交价格权重越大
# where
# t = a trading day
# V = the daily turnover
# P = the stock price
# R = the reference price
# CGO =（P(t-5)-R(t-5)）/P(t-5) # (t-5)为下标表示时间
# CGO表示现在价格与之前一段时间成交量加权平均价格的偏差
# 这里R用前三个月的的换手率加权平均值
def get_df_CGO(stock_list, context, asc):
    # 上个交易日日期
    yest = context.previous_date
    # 5天前
    date_5days_before = shift_trading_day(yest, shift = -5)
    str_5days_before=str(date_5days_before)
    # 3个shift前的交易日日期
    date_3shift_before = shift_trading_day(yest, shift = -61)
    str_3shift_before=str(date_3shift_before)
    # 3个月前至5天前的均价
    panel_price_3shift = get_price(list(stock_list), 
                       start_date=date_3shift_before, 
                       end_date=date_5days_before, 
                       frequency='daily', 
                       fields=['avg','volume'])
    # 利用tushare获取股票换手率信息 也不行
    # # 股票代码格式转换
    # def stock_type(stock_list):
    #     stock_list=list(stock_list)
    #     stock_num=list()
    #     for stock in stock_list:
    #         temp=stock[:6]
    #         temp=str(temp)
    #         stock_num.append(temp)
    #     return stock_num
    # list_stock = stock_type(stock_list)   
    # temp=ts.get_h_data('600848')
    
    # 获取流通市值
    df_CMC=get_fundamentals(query(valuation.code, valuation.circulating_cap
                     ).filter(valuation.code.in_(stock_list)), date=yest)
    df_CMC.index=df_CMC.code
    del df_CMC['code']
    df_CMC=df_CMC.T
    # 计算3个月前至5天前的每天换手率
    df_TR=DataFrame(columns=panel_price_3shift['volume'].columns)
    temp_L=len(panel_price_3shift['volume'])
    for i in range(temp_L):
        df_temp=panel_price_3shift['volume'].ix[i,:]/(df_CMC*10000)
        df_TR=df_TR.append(df_temp)
    # 计算R(t-5)值
    def get_df_R(df_price,df_TR):
        L=len(df_price)
        df_R=DataFrame(columns=df_price.columns)
        df_temp=df_price.ix[0,:]
        df_R=df_R.append(df_temp)
        for i in range(1,L):
            df_temp=df_TR.ix[i,:]*df_price.ix[i,:]+(1-df_TR.ix[i,:])*df_R.ix[i-1,:]
            df_R=df_R.append(df_temp,ignore_index=True)
        return df_R  
    df_price=panel_price_3shift['avg']
    df_R=get_df_R(df_price,df_TR)
    # 取前5天那一天的值
    df_R=df_R.ix[temp_L-1,:]
    df_price=df_price.ix[temp_L-1,:]
    # CGO值
    series_CGO=(df_price-df_R)/df_price
    # 生成DataFrame
    df_CGO=DataFrame(series_CGO)
    df_CGO.columns=['CGO']
    # 删除NaN
    df_CGO = df_CGO.dropna()
    # print df_CGO
    # 排序给出排序打分
    df_CGO['CGO_sorted_rank'] = df_CGO['CGO'].rank(ascending = asc, method = 'dense')
    return df_CGO
    
# 6 TO 流通股本日均换手率
def get_df_TO(stock_list, context, asc):
    # 获取价格数据,当前到21天前一共22行，与之前get_price不同，没有使用转置，行为股票代码
    # 列为日期，上边为较早最后为较晚
    df_volume = get_price(list(stock_list), 
                       start_date=context.previous_date, 
                       end_date = context.previous_date, 
                       frequency = 'daily', 
                       fields = 'volume')['volume']
    # 获得换手率(%)turnover_ratio
    df_TO = get_fundamentals(query(valuation.code, valuation.circulating_cap
                     ).filter(valuation.code.in_(stock_list)), context.previous_date)
    # # 删除nan
    # 使用股票代码作为index
    df_TO.index = df_TO.code
    # 去除没有流通市值的成交量
    list_temp=list(df_TO.index)
    df_volume=df_volume.ix[0,list_temp]
    # 生成TO
    df_TO['TO'] = df_volume/ (df_TO['circulating_cap'] * 10000)
    # 删除无用数据
    del df_TO['code']
    del df_TO['circulating_cap']
    #删除NaN
    df_TO=df_TO.dropna()
    # 生成排名序数
    df_TO['TO_sorted_rank'] = df_TO['TO'].rank(ascending = asc, method = 'dense')
    return df_TO

# 7 EPS_GR(EPS Growth Ratio)每股收益增长率（年度同比）
def get_df_EPS_GR(stock_list, context, asc):
    # 获取日期
    yest = context.previous_date
    date_oneyear_before = shift_trading_day(yest, shift = -243)
    # 查询财务数据
    df_EPS = get_fundamentals(query(indicator.code,  indicator.eps
                     ).filter(indicator.code.in_(stock_list)), date=yest)
    df_EPS.index=df_EPS.code
    series_EPS=df_EPS['eps']
    df_EPS_oneyear_before = get_fundamentals(query(indicator.code,  indicator.eps
                     ).filter(indicator.code.in_(stock_list)), date=date_oneyear_before)
    df_EPS_oneyear_before.index=df_EPS_oneyear_before.code
    series_EPS_oneyear_before=df_EPS_oneyear_before['eps']
    # 计算每股收益增长率DPS_GR 
    df_EPS_GR=DataFrame(columns=['EPS_GR'])
    df_EPS_GR['EPS_GR'] = (series_EPS - series_EPS_oneyear_before)/series_EPS
    #删除NaN
    df_EPS_GR=df_EPS_GR.dropna()
    # 生成排名序数
    df_EPS_GR['EPS_GR_sorted_rank'] = df_EPS_GR['EPS_GR'].rank(ascending = asc, method = 'dense')
    return df_EPS_GR
    

# 8 Revenue Growth Ratio营业收入增长率 季度同比 
def get_df_RGR(stock_list, context, asc):
    # 获取日期
    yest = context.previous_date
    control_date = 0
    date_num = -63
    while 1:
        date_oneseason_before = shift_trading_day(yest, shift = date_num)
        # 查询财务数据
        df_Revenue = get_fundamentals(query( income.code, income.operating_revenue
                         ).filter( income.code.in_(stock_list)), date=yest)
        df_Revenue.index=df_Revenue.code
        series_Revenue=df_Revenue['operating_revenue']
        df_Revenue_oneseason_before = get_fundamentals(query( income.code, income.operating_revenue
                         ).filter( income.code.in_(stock_list)), date=date_oneseason_before)
        df_Revenue_oneseason_before.index=df_Revenue_oneseason_before.code
        series_Revenue_oneseason_before=df_Revenue_oneseason_before['operating_revenue']
        # 计算营业收入增长率 
        df_RGR=DataFrame(columns=['RGR'])
        df_RGR['RGR'] = (series_Revenue - series_Revenue_oneseason_before)/series_Revenue
        #删除NaN
        df_RGR=df_RGR.dropna()
        # 生成排名序数
        df_RGR['RGR_sorted_rank'] = df_RGR['RGR'].rank(ascending = asc, method = 'dense')
        control_date = len([ i for i in list(df_RGR['RGR']) if i > 0.0 or i < 0.0])
        date_num -= 10
        if control_date > 0.9*len(list(df_RGR['RGR'])):
            return df_RGR


# 9 净利润增长率季度同比 net_profit Growth Ratio
def get_df_NPGR(stock_list, context, asc):
    yest = context.previous_date
    control_date = 0
    date_num = -63
    while 1:
        date_oneseason_before = shift_trading_day(yest, shift = date_num)
        # 查询财务数据
        df_NP = get_fundamentals(query( income.code, income.net_profit
                         ).filter( income.code.in_(stock_list)), date=yest)
        df_NP.index=df_NP.code
        series_NP=df_NP['net_profit']
        df_NP_oneseason_before = get_fundamentals(query( income.code, income.net_profit
                         ).filter( income.code.in_(stock_list)), date=date_oneseason_before)
        df_NP_oneseason_before.index=df_NP_oneseason_before.code
        series_NP_oneseason_before=df_NP_oneseason_before['net_profit']
        # 计算净利润增长率
        df_NPGR=DataFrame(columns=['NPGR'])
        df_NPGR['NPGR'] = (series_NP - series_NP_oneseason_before)/series_NP
        #删除NaN
        df_NPGR=df_NPGR.dropna()
        # 生成排名序数
        df_NPGR['NPGR_sorted_rank'] = df_NPGR['NPGR'].rank(ascending = asc, method = 'dense')
        control_date = len([ i for i in list(df_NPGR['NPGR']) if i > 0.0 or i < 0.0])
        date_num -= 10
        if control_date > 0.9*len(list(df_NPGR['NPGR'])):
            return df_NPGR

# 10 非流动性因子illiquidity factor(ILLIQ) 
# 市场的流动性越差价格让步越大 越能获得超额收益 但是这个因子受小市值影响明显
# ILLIQ=(1/N)sum(abs(Ri)/Vi) 每日价格变化幅度绝对值和成交额的比值求平均 这里N取5，即过去5天求平均
def get_df_ILLIQ(stock_list, context, asc):
    yest = context.previous_date
    date_5days_before = shift_trading_day(yest, shift = -4) # 包含yest以及之前的4天数据共5天
    date_6days_before = shift_trading_day(yest, shift = -5)
    # 获取涨跌幅信息 避免时间差引起的股票不一致获取六天数据 去掉最新的一天
    df_volume = get_price(list(stock_list), 
                       start_date=date_6days_before, 
                       end_date = yest, 
                       frequency = 'daily', 
                       fields = ['close'])['close']
    df_volume=df_volume.ix[:5,:]
    # 获取成交量信息
    panel_volume = get_price(list(stock_list), 
                       start_date=date_5days_before, 
                       end_date = yest, 
                       frequency = 'daily', 
                       fields = ['volume','close'])
    # 计算涨跌幅 坑爹的聚宽API不能获取一段时间的涨跌幅 只好自己计算
    df_volume.index=panel_volume['close'].index
    panel_volume['change']=(panel_volume['close']-df_volume)/df_volume
    # # 涨跌幅 用于多因子 
    # series_price_change=1.0*panel_volume['change'].ix[-1,:].T
    # series_price_change=series_price_change.dropna()
    # 绝对值
    panel_volume['change']=panel_volume['change'].abs()
    temp=(panel_volume['change']/panel_volume['volume'])
    # 成交量单位换成亿元
    panel_volume['volume']=panel_volume['volume']/100000000
    df_ILLIQ=DataFrame()
    df_ILLIQ['ILLIQ']=(panel_volume['change']/panel_volume['volume']).sum().T*0.2
    df_ILLIQ=df_ILLIQ.dropna()
    # 生成排名序数
    df_ILLIQ['ILLIQ_sorted_rank'] = df_ILLIQ['ILLIQ'].rank(ascending = asc, method = 'dense')
    return df_ILLIQ #, series_price_change

#11 按照Fama-French规则计算k个参数并且回归（三因子或五因子模型），计算出股票的alpha并且输出DataFrame
def get_df_FF (stock_list, context, asc):
    # 三因子NoF=3，五因子NoF=5
    NoF=3
    # 无风险利率
    rf=0.04
    # 时间
    yest = context.previous_date
    date_3month_before = shift_trading_day(yest, shift = -61)
    date_1year_before = shift_trading_day(yest, shift = -243)
    # 股票个数
    LoS=len(stock_list)
    #查询三因子/五因子的语句
    q = query(
        valuation.code,
        valuation.market_cap,
        (balance.total_owner_equities/valuation.market_cap/100000000.0).label("BTM"),
        indicator.roe,
        balance.total_assets.label("Inv")
    ).filter(valuation.code.in_(stock_list))
    df = get_fundamentals(q,yest)
    #计算5因子再投资率的时候需要跟一年前的数据比较，所以单独取出计算
    ldf=get_fundamentals(q,date_1year_before)
    # 若前一年的数据不存在，则暂且认为Inv=0
    if len(ldf)==0:
        ldf=df
    df["Inv"]=np.log(df["Inv"]/ldf["Inv"])
    # 选出特征股票组合
    S=df.sort('market_cap')['code'][:LoS/3]
    B=df.sort('market_cap')['code'][LoS-LoS/3:]
    L=df.sort('BTM')['code'][:LoS/3]
    H=df.sort('BTM')['code'][LoS-LoS/3:]
    W=df.sort('roe')['code'][:LoS/3]
    R=df.sort('roe')['code'][LoS-LoS/3:]
    C=df.sort('Inv')['code'][:LoS/3]
    A=df.sort('Inv')['code'][LoS-LoS/3:]
    # 获得样本期间的股票价格并计算日收益率，
    df2 = get_price(list(stock_list),date_3month_before,yest,'1d')
    df3=df2['close'][:]
    #取自然对数再差分求收益率（涨跌幅）的近似
    df4=np.diff(np.log(df3),axis=0)+0*df3[1:] 
    #求因子的值
    SMB=sum(df4[S].T)/len(S)-sum(df4[B].T)/len(B)
    HMI=sum(df4[H].T)/len(H)-sum(df4[L].T)/len(L)
    RMW=sum(df4[R].T)/len(R)-sum(df4[W].T)/len(W)
    CMA=sum(df4[C].T)/len(C)-sum(df4[A].T)/len(A)
    #用沪深300作为大盘基准
    dp=get_price('000001.XSHG',date_3month_before,yest,'1d')['close']
    #取自然对数再差分求收益率（涨跌幅）的近似
    RM=diff(np.log(dp))-rf/243
    #将因子们计算好并且放好
    X=pd.DataFrame({"RM":RM,"SMB":SMB,"HMI":HMI,"RMW":RMW,"CMA":CMA})
    #取前NoF个因子为策略因子
    factor_flag=["RM","SMB","HMI","RMW","CMA"][:NoF]
    # print X
    X=X[factor_flag]
    # 线性回归函数
    def linreg(X,Y):
        X=sm.add_constant(array(X))
        Y=array(Y)
        if len(Y)>1:
            results = regression.linear_model.OLS(Y, X).fit()
            # 这里输出
            return results.rsquared
        else:
            return [float("nan")]
    # 对样本数据进行线性回归并计算alpha
    t_scores=[0.0]*LoS
    for i in range(LoS):
        t_stock=stock_list[i]
        # sample=pd.DataFrame()
        t_r=linreg(X,df4[t_stock]-rf/243)
        t_scores[i]=t_r
    # 这个scores就是alpha 
    df_FF=pd.DataFrame({'FF':t_scores})
    df_FF.index=stock_list
    # 去掉缺失的值
    df_FF=df_FF.dropna()
    # 生成排名序数
    df_FF['FF_sorted_rank'] = df_FF['FF'].rank(ascending = asc, method = 'dense')
    return df_FF

# 12 流通市值
def get_df_CMC(stock_list, context, asc):
    # 获得流通市值 circulating_market_cap 流通市值(亿)
    df_CMC = get_fundamentals(query(valuation.code, valuation.circulating_market_cap
                     ).filter(valuation.code.in_(stock_list)))
    df_CMC.index=df_CMC['code']
    # 删除nan
    df_CMC = df_CMC.dropna()
    df_CMC['CMC']=df_CMC['circulating_market_cap']
    # 删除无用信息
    del df_CMC['circulating_market_cap']
    del df_CMC['code']
    # 生成排名序数
    df_CMC['CMC_sorted_rank'] = df_CMC['CMC'].rank(ascending = asc, method = 'dense')
    return df_CMC

# **********************多因子选股************************
# 因子去极值
def Remove_Extremum(df_column):
    #单列DataFrame 或者Series
    # import pandas as pd
    # import numpy as np
    import math
    Te_data=df_column.values
    # index=np.array(df_column.index)
    md=np.median(Te_data)
    md_up=[]
    md_down=[]
    #计算mc
    for i in Te_data:
        if i>md:
            md_up.append(i)
        if i<md:
            md_down.append(i)
    container=[]
    for i in md_up:
        for j in md_down:
            container.append(((i-md)-(md-j))/(i-j))
    mc=np.median(np.array(container))
    #计算L U
    Q1=np.percentile(Te_data,25)
    Q3=np.percentile(Te_data,75)
    IQR=Q3-Q1
    if mc>=0:
        L=Q1-1.5*math.exp(-3.5*mc)*IQR
        U=Q3+1.5*math.exp(4*mc)*IQR
    else:
        L = Q1 - 1.5 * math.exp(-4 * mc) * IQR
        U = Q3 + 1.5 * math.exp(3.5 * mc) * IQR
    #输出结果
    Lr=len(df_column)
    DropIndex=[]
    for i in range(0,Lr,1):
        if df_column.iloc[i]>U or df_column.iloc[i]<L:
            DropIndex.append(i)
    temp=df_column[DropIndex].index
    dorp_labels=list(temp)
    df_result=df_column.drop(dorp_labels)
    return df_result

# 计算IC值
def cal_IC(df_AllFactor,price_change):
    # 计算每一个因子与收益的IC值
    # df_AllFactor为DataFrame每一个columns为对应因子名称
    # price_change为DataFrame或者series为股票下一期的收益率
    # 计算排名序数
    df_AllFactor = df_AllFactor.dropna()
    # 生成装排名的DataFrame
    df_AllFactor_rank=0.0*df_AllFactor
    for column in df_AllFactor.columns:
        df_AllFactor_rank[column]=df_AllFactor[column].rank(ascending = False, method = 'dense')
    # 添加涨跌幅排名
    price_change=price_change.dropna()
    df_AllFactor_rank['price_change']=price_change.rank(ascending = False, method = 'dense')
    # 计算IC值
    df_AllFactor_rank=df_AllFactor_rank.dropna()
    list_IC=[]
    for column in df_AllFactor.columns:
        temp_IC=np.corrcoef(df_AllFactor_rank[column],df_AllFactor_rank['price_change'])[0,1]
        list_IC.append(temp_IC)
    df_IC=DataFrame(data=list_IC,index=df_AllFactor.columns)
    return df_IC

# 线性回归选股函数
def linear_regression_choose(stock_list, context, asc):
    # 时间
    yest = context.previous_date
    # 获取去极值后的因子值
    for name in g.list_factor_name:
        temp_a='get_df_'+name
        temp_b='series_'+name
        # 获取因子值
        temp_series=globals()[temp_a](stock_list, context, asc)[name]
        # 去极值
        temp_series=Remove_Extremum(temp_series)
        # 结果存入series变量
        globals()[temp_b]=temp_series
    # 因子汇总 并保存到全局变量
    # 将因子所含股票代码取并集
    set_temp=set(globals()[g.list_index_name[0]].index)
    for i in range(len(g.list_index_name)):
        if i>0:
            set_temp=set_temp | set(globals()[g.list_index_name[i]].index)
        else:
            pass
    list_temp_index=list(set_temp)
    # ******未标准化因子存储*******
    # 定义DataFrame作为这一期所有因子以及涨跌幅的容器（去极值后的因子值，未标准化）
    df_AllFactor_value=DataFrame(index=list_temp_index)
    # 将因子值写入DataFrame
    for columns_name in g.list_index_name:
        df_AllFactor_value[columns_name]=globals()[columns_name]
    # 将columns换名字
    df_AllFactor_value.columns=g.list_factor_name
    # 获取这一期一个月涨跌幅
    df_AllFactor_value['price_change']=series_MTM1
    # 写入全局变量
    str_date=str(yest)
    panel_temp=pd.Panel({str_date:df_AllFactor_value})
    g.panel_factor_value=pd.concat([g.panel_factor_value,panel_temp])
    # *******************    
    # 因子线性回归系数加权
    df_AllFactor_value=df_AllFactor_value[g.list_factor_name]
    # 因子值按线性回归结果加权
    # 各因子权重
    series_linear_weight=Series([-0.130625,
                                -0.2776,
                                -0.895675,
                                2.237025,
                                -0.4901,
                                -5.46015,
                                0.3308,
                                -0.1371,
                                0.07575,
                                0.1363,
                                -0.635675,
                                0.002425])
    series_linear_weight.index=g.list_factor_name
    # 加权后的因子值
    df_AllFactor_value=df_AllFactor_value*series_linear_weight
    # 因子值求和 这里格式变为Series
    df_AllFactor_value=df_AllFactor_value.sum(axis=1)
    # 格式变为DataFrame并命名因子
    df_AllFactor=DataFrame(df_AllFactor_value)
    df_AllFactor.columns=['AllFactor']
    # 去掉空值
    df_AllFactor=df_AllFactor.dropna()
    # 行业中性化
    if g.Industry_Neutral:
        df_AllFactor = INDUSTRY_SORTED(stock_list,df_AllFactor,'AllFactor')
    # 求排名序数
    df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending = asc, method = 'dense')
    return df_AllFactor
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    '''
    # 计次
    g.count=g.count+1
    # 线性回归得到回归系数
    if g.count > g.periods:
        # 线性回归取得因子权重
        # 取一个periods收益率最高的1/3股票回归因子系数
        series_cumulate_profit = g.panel_factor_value.ix[1:,:,-1].sum(axis=1)
        # 收益排序
        series_cumulate_profit = series_cumulate_profit.sort(ascending=False,inplace=False)
        # 取收益最高1/3股票
        series_selected = series_cumulate_profit[:len(series_cumulate_profit)//3]
        # 每一个因子均取收益最高1/3股票
        g.panel_factor_value = g.panel_factor_value.ix[:,series_selected.index,:]

        # **************
        # 取得因子值的均值 不要‘price_change’，格式：columns为因子名称，index为时期
        df_factor_mean=g.panel_factor_value.ix[:-1,:,:-1].mean().T
        # 取得收益‘price_change’
        series_price_mean=g.panel_factor_value.ix[1:,:,-1].mean()
        # 调整index
        series_price_mean.index=df_factor_mean.index
        # 线性回归
        X=sm.add_constant(df_factor_mean)
        results = regression.linear_model.OLS(series_price_mean,X).fit()
        # 因子的回归系数
        print results.summary()   
    # else:
        # 这里随便取因子，以免报错，不会实际交易
        df_AllFactor=DataFrame({'AllFactor':series_MTM1})
        df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending = asc, method = 'dense')
        df_AllFactor.columns=['AllFactor','AllFactor_sorted_rank']
    else:
        # 这里随便取因子，以免报错，不会实际交易
        df_AllFactor=DataFrame({'AllFactor':series_MTM1})
        df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending = asc, method = 'dense')
        df_AllFactor.columns=['AllFactor','AllFactor_sorted_rank']
        return df_AllFactor
    '''

# 最优化选股函数
def optimize_choose(stock_list, context, asc):
    # 计次
    g.count=g.count+1
    # 时间
    yest = context.previous_date
    # 获取去极值后的因子值
    for name in g.list_factor_name:
        temp_a='get_df_'+name
        temp_b='series_'+name
        # 获取因子值
        temp_series=globals()[temp_a](stock_list, context, asc)[name]
        # 去极值
        temp_series=Remove_Extremum(temp_series)
        # 结果存入series变量
        globals()[temp_b]=temp_series
    # 因子汇总 并保存到全局变量
    # list_index=g.list_index_name
    # 将因子所含股票代码取并集
    set_temp=set(globals()[g.list_index_name[0]].index)
    for i in range(len(g.list_index_name)):
        if i>0:
            set_temp=set_temp | set(globals()[g.list_index_name[i]].index)
        else:
            pass
    list_temp_index=list(set_temp)
    # 最优化选股
    # ******标准化后因子存储**********
    # 数据标准化函数
    def Z_ScoreNormalization(x):
        # import numpy as np
        x = x.dropna()
        x = (x - np.average(x)) / np.std(x)
        return x
    # 生成DataFrame（去极值、标准化后的因子值）
    df_AllFactor=DataFrame(index=list_temp_index)
    # 将因子值标准化并写入DataFrame
    for columns_name in g.list_index_name:
        temp=globals()[columns_name]
        temp=Z_ScoreNormalization(temp)
        df_AllFactor[columns_name]=temp
    # 将columns换名字
    df_AllFactor.columns=g.list_factor_name
    # 获取涨跌幅
    df_AllFactor['price_change']=Z_ScoreNormalization(series_MTM1[list_temp_index])
    # 写入全局变量
    str_date=str(yest)
    panel_temp=pd.Panel({str_date:df_AllFactor})
    g.panel_factor=pd.concat([g.panel_factor,panel_temp])
    # *******************
    # 计数
    temp_count=len(g.panel_factor.items)
    # 计算Rank IC    
    if temp_count>1:
        # 因子值
        df_all_factor=g.panel_factor.ix[-2,:,:]
        # 涨跌幅
        series_price_change=g.panel_factor.ix[-1,:,:]['price_change']
        g.df_IC[str_date]=cal_IC(df_all_factor,series_price_change)
        # 计算相关系数矩阵 g.periods
        temp_count=len(g.df_IC.T)
        if temp_count>g.periods-1:
            df_temp=g.df_IC.ix[:,-g.periods:]
            df_temp=df_temp.dropna()
            array_IC_mean=array(df_temp.mean(axis=1))
            array_corr=np.corrcoef(df_temp)
    # 最优化 计算权重向量
    import scipy.optimize as sco
    def statistics(weights):
        weights = np.array(weights)
        ic_mean = array_IC_mean
        ic_cov = array_corr
        port_ir = np.dot(weights.T,ic_mean) / np.sqrt(np.dot(weights.T,np.dot(ic_cov,weights)))
        return port_ir
    def min_ir(weights):
        return -statistics(weights)
    def port_weight():
        noa = len(g.list_index_name)
        #约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
        # cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
        #我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
        bnds = tuple((-1,1) for x in range(noa))
        #优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。, constraints = cons
        optv = sco.minimize(min_ir, noa*[1./noa,],method = 'SLSQP', bounds = bnds)
        # 返回权重向量
        return optv['x'].round(3)
    if g.count > g.periods:
        weight_vector = port_weight()
        # 生成series
        series_weight=Series(data=weight_vector,index=g.list_factor_name)
        df_temp=g.panel_factor.ix[-1,:,:].dropna()
        # 去掉price_change
        del df_temp['price_change']
        # 加权后的因子值
        df_temp=df_temp*series_weight
        # 清空DataFrame
        df_AllFactor=DataFrame()
        df_AllFactor['AllFactor']=df_temp.T.sum()
        # 行业中性化
        if g.Industry_Neutral:
            df_AllFactor = INDUSTRY_SORTED(stock_list,df_AllFactor,'AllFactor')
        df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending = asc, method = 'dense')
        return df_AllFactor
    else:
        # 这里随便取因子，以免报错，不会实际交易
        df_AllFactor=DataFrame({'AllFactor':series_MTM1})
        df_AllFactor['AllFactor_sorted_rank'] = df_AllFactor['AllFactor'].rank(ascending = asc, method = 'dense')
        df_AllFactor.columns=['AllFactor','AllFactor_sorted_rank']
        return df_AllFactor

# 东方证券机器学习因子库因子，集成到一起
def get_df_AllFactor(stock_list, context, asc):
    # 线性回归选股
    if g.linear_regression:
        df_AllFactor=linear_regression_choose(stock_list, context, asc)
    # 最优化选股
    if g.optimal_choose:
        df_AllFactor=optimize_choose(stock_list, context, asc)
    return df_AllFactor

    
'''
================================================================================
每天收盘后
================================================================================
'''
# 每日收盘后要做的事情（本策略中不需要）
def after_trading_end(context):
    return