import pandas as pd
import numpy as np
import matplotlib as plt


# ================== 1 Object Creation===================
# series
# s = pd.Series([1,3,5,np.nan,6,8])
# print(s)
# dataframe
# dates = pd.date_range('20130101', periods=6)
# 创建 6行 4列的 np数组，指定index，和表头
# df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
# s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
# df['F'] = s1
# print(df)
# 手动创建dataframe，key为表头，value为其对应的一列
# df2 = pd.DataFrame({ 'A' : 1.,
#                      'B' : pd.Timestamp('20130102'),
#                      'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
#                      'D' : np.array([3] * 4,dtype='int32'),
#                      'E' : pd.Categorical(["test","train","test","train"]),
#                      'F' : 'foo' })




# ================= 2 Viewing Data ======================
# print(df2)
# 获取每一列的数据类型
# print(df2.dtypes)
# 获取 dataframe 的指定行数
# print(df2.head())
# print(df2.tail(3))
# 索引
# print(df.index)
# 表头
# print(df.columns)
# 数据
# print(df.values)
# 显示您的数据的快速统计摘要
# print(df.describe())
# 转置dataframe数据
# print(df.T)
# 按表头排序，降序，（列排序）
# print(df.sort_index(axis=1, ascending=False))
# 按某一列的值进行行排序,（行排序）
# print(df.sort_values(by='B'))
# 选择一个列，产生一个Series，相当于df.A





# ================== 3 Selection ===================
# print(df['A'])
# print(df.A)
# 进行行切片
# print(df[0:3])
# 单标签选择多轴

# ---------- Selection by Label -----------
# print(df.loc[dates[0]])
# 多标签选择多轴
# print(df.loc[:,['A','B']])
# 标签与轴的切片
# print(df.loc['20130102':'20130104',['A','B']])
# print(df.loc['20130102',['A','B']])
# 精确取到某个数据
# print(df.loc[dates[0],'A'])
# print(df.at[dates[0],'A'])

# ------------Selection by Position----------
# print(df.iloc[3])
# print(df.iloc[3:5,0:2])
# print(df.iloc[[1,2,4],[0,2]])
# print(df.iloc[1:3,:])
# print(df.iloc[:,1:3])
# print(df.iloc[1,1])
# print(df.iat[1,1])

# ---------------Boolean Indexing---------------
# print(df[df.A > 0])
# print(df[df > 0])
# df2 = df.copy()
# df2['E'] = ['one', 'one','two','three','four','three']
# print(df2)
# print(df2[df2['E'].isin(['two','four'])])

# ---------------setting after selection------------
# df.at[dates[0],'A'] = 0
# 按位置设置
# df.iat[0,1] = 0
# df.loc[:,'D'] = np.array([5] * len(df))
# df2 = df.copy()
# df2[df2 > 0] = -df2
# print(df2)





#====================== 4 Missing Data =======================
# 使用reindex来增删改查列
# df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
# df1.loc[dates[0]:dates[1],'E'] = 1
# print(df1)
# 删除有空的行,以返回的形式，不改变原对象
# a = df1.dropna(how='any')
# a = df1.fillna(value=5)
# a = pd.isna(df1)
# print(a)




# ==================== 5 Operations =========================
# 获取平均值
# print(df.mean())
# 获取每行的平均值
# print(df.mean(1))
# 左偏两个位置，pandas会根据index，空的位置用nan填充
# s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
# print(s)
# 减法操作，有nan时该位置为nan
# a = df.sub(s, axis='index')
# print(a)

# -------------apply function --------
# 对于每一列进行操作
# print(df.apply(np.cumsum))
# print(df.apply(lambda x: x.max() - x.min()))

# ---------------Histogramming------------
# s = pd.Series(np.random.randint(0, 7, size=10))
# print(s)
# # 统计
# print(s.value_counts())

# --------------String Methods--------------
# s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
# s= s.str.lower()
# print(s)






# ========================= 6 Merge ==========================
# df = pd.DataFrame(np.random.randn(10, 4))
# print(df)

# ------------Concat------------
# pieces = [df[:3], df[3:7], df[7:]]
# print(pieces)
# a = pd.concat(pieces)
# print(a)

# ------------Join--------------
# left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
# right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
# # print(left)
# print(right)
# a = pd.merge(left, right, on='key')
# print(a)
# left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
# right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
# a = pd.merge(left, right, on='key')
# print(a)

# -------------append-------------
# df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
# print(df)
# s = df.iloc[3]
# print(s)
# a = df.append(s, ignore_index=True)
# print(a)




# ====================== 7 Grouping ========================
# df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
#                           'foo', 'bar', 'foo', 'foo'],
#                    'B' : ['one', 'one', 'two', 'three',
#                           'two', 'two', 'one', 'three'],
#                    'C' : np.random.randn(8),
#                    'D' : np.random.randn(8)})

# print(df)
# print(df.groupby('A').sum())
# print(df.groupby(['A','B']).sum())






# ======================= 8 Reshaping ========================

# --------------------Stack---------------------
# tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
#                      'foo', 'foo', 'qux', 'qux'],
#                     ['one', 'two', 'one', 'two',
#                      'one', 'two', 'one', 'two']]))
# # print(tuples)
# index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
# df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
# print(df)
# df2 = df[:4]
# print(df2)
# # 堆叠成树状
# stacked = df2.stack()
# print(stacked)
# b = stacked.unstack(1)
# print(b)

# ---------------------Pivot Tables------------------
# df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
#                    'B' : ['A', 'B', 'C'] * 4,
#                    'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
#                    'D' : np.random.randn(12),
#                    'E' : np.random.randn(12)})
# print(df)
# a = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
# print(a)





# ====================== 9 Time Series=======================
# rng = pd.date_range('1/1/2012', periods=100, freq='S')
# print(rng)
# ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
# print(ts)
# a = ts.resample('5Min').sum()
# print(a)
# rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
# ts = pd.Series(np.random.randn(len(rng)), rng)
# print(ts)
# 切换市区
# ts_utc = ts.tz_localize('US/Eastern')
# print(ts_utc)
# 切换时间的展示方式
# rng = pd.date_range('1/1/2012', periods=5, freq='M')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
# 显示月份
# ps = ts.to_period()
# print(ps)
# 切换回日期
# ps = ps.to_timestamp()
# print(ps)
# prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
# print(prng)
# ts = pd.Series(np.random.randn(len(prng)), prng)
# ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
# print(ts.head())





# ========================== 10 Categoricals ==========================
# df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
# 新加一个列并更换数据类型
# print(df)
# 可以看到此时类标签的字符a,b,e ；类标签的集合为[a,b,e] 
# 打印类标签的结果如下 
# df["grade"] = df["raw_grade"].astype("category")
# print(df['grade'])
# df["grade"].cat.categories = ["very good", "good", "very bad"]
# print(df['grade'])
# 给categories赋值，可以改变类别标签。赋值的时候是按照顺序进行对应的。a对应very good，b对应good,c对应very bad。操作完成之后，原来的标签a就变成了very good标签。 
# 此时类标签的集合为[“very good”, “good”, “very bad”]
# 改变类别标签集合，操作过后数据的标签不变，但是标签的集合变为[“very bad”, “bad”, “medium”, “good”, “very good”]
# df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
# print(df['grade'])
# print(df.sort_values(by="grade"))








