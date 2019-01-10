import pandas as pd
import numpy as np

import time


#设置打印展示输出
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width',10000)

#读取json数据


class report:


	def __init__(self):
		self.bill = pd.read_json('bill.json',encoding='utf-8')
		self.base_info = pd.read_json('base_info.json',encoding='utf-8')
		self.call_record = pd.read_json('call_record.json',encoding='utf-8')
		self.call_record['call_date'] = pd.to_datetime(self.call_record['call_date'])
		self.report = {}
		print(self.call_record[:1])
		# print(self.call_record['call_type']==2)

		# print(base_info)
		# print(bill)


	@staticmethod
	def to_json_by_rows(df):
		return df.to_json(lines=True,orient='records')

	@staticmethod
	def to_python_obj_by_rows(df):
		return df.to_dict(orient='record')

	@staticmethod
	def call_contact_detail(call_record):
		def handle(group):
			return pd.Series({
				'peer_num':group.name,
				'city':group['other_area'][group.index[0]] or '全国',
				'call_cnt_total':group.shape[0],
				'call_cnt_1w':group[(pd.to_datetime('2018-12') > group['call_date']) &( group['call_date'] > pd.to_datetime('2018-8'))].shape[0]
				})

		df = call_record.groupby('other_mobile').apply(handle)

		print(df)



	@staticmethod
	def contact_area_data(call_record):
		def handle(group):
			return pd.Series({
				'area':group.name,
				'call_mobile_num':group['other_mobile'].nunique(),
				'call_cnt':group.shape[0],
				'caller_cnt':group[group['call_type']==2].shape[0],
				'called_cnt':group[group['call_type']==1].shape[0],
				'call_time':group['call_long_hour'].sum(),
				'caller_call_time':group[group['call_type']==2]['call_long_hour'].sum(),
				'called_call_time':group[group['call_type']==1]['call_long_hour'].sum(),
				})

		df = call_record.groupby('other_area').apply(handle)
		return report.to_python_obj_by_rows(df)


	@staticmethod
	def trip_record(call_record):
		result = []
		df = call_record.sort_values(by='call_date')
		ext =df.loc[df.index[0]]
		for _ in df.itertuples():
			if _.call_area != ext.call_area:
				item =dict(departure=ext.call_area,destination=_.call_area,
					departure_time=ext.call_date,destination_time=_.call_date,
					day_type='双休日' if pd.to_datetime(_.call_date).dayofweek in [5,6] else '工作日'
					)
				ext = _
				result.append(item)
		return result


	@staticmethod
	def action_watch(call_record):	
		df = call_record.groupby('other_mobile').agg({
			'other_area':lambda x:x,
			'other_mobile':lambda x:x.size
			}).rename(columns={'other_area': 'attribution',
    	                       'other_mobile': 'call_cnt'})
		return report.to_python_obj_by_rows(df)



	def gen_report(self):
		a = time.time()
		call_contact_detail = self.call_contact_detail(self.call_record)
		print(time.time()-a)
		# contact_area_data = self.contact_area_data(self.call_record)
		# print(contact_area_data)
		# trip_record = self.trip_record(self.call_record)
		# action_watch = self.action_watch(self.call_record)



if __name__ == '__main__':
	obj = report()
	obj.gen_report()






