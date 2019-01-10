import datetime
import json
import re
from collections import defaultdict

import numpy
import pandas as pd
from pandas import DataFrame

from utils.constant import 会话发起方式, 会话结束方式


def is_df_empty(df, tables):
    return all(df[table].shape[0] == 0 for table in tables)


def default(o):  # numpy.int64 无法json化，因此在json.dumps时，要将其转化为int
    if isinstance(o, numpy.integer):
        return int(o)


class BaseStatis(object):
    def __init__(self, df, statis_type=1):
        self.df = df
        self.statis = dict()
        self.statis_type = statis_type
        if "question_tag" in self.df:
            self.tags = self._get_tags
        else:
            self.tags = {}

    def _table(self, table):
        if table in self.df:
            return self.df[table]
        else:
            return DataFrame({"id": []})

    def _series_index(self, series, idx):
        """
        根据索引获取一个Series对象的值
        """
        if series.size > 0:
            return series.iloc[idx]
        else:
            return {}

    def _get_today_date(self, now, start='start'):
        if start == 'start':
            if now:
                return datetime.datetime.combine(
                    datetime.date.today(),
                    datetime.time.min) - datetime.timedelta(
                    days=1)
            else:
                statistic_now = datetime.datetime.now() - datetime.timedelta(hours=1)
                year = statistic_now.year
                month = statistic_now.month
                day = statistic_now.day
                hour = statistic_now.hour
                return datetime.datetime(year=year, day=day, month=month,
                                         hour=hour)

        elif start == 'end':
            if now:
                return datetime.datetime.combine(
                    datetime.date.today(),
                    datetime.time.max) - datetime.timedelta(
                    days=1)
            else:
                statistic_hour = datetime.datetime.now()
                year = statistic_hour.year
                month = statistic_hour.month
                day = statistic_hour.day
                hour = statistic_hour.hour
                return datetime.datetime(year=year, day=day, month=month,
                                         hour=hour) - datetime.timedelta(
                    seconds=1)

    def _count(self, ser):
        return int(ser.id.count())

    @property
    def _get_tags(self):
        return {
            _tag.id: _tag.name for _tag in self.df["question_tag"].itertuples()
        }

    def _json_dumps(self, d):
        return json.dumps(d, ensure_ascii=False, default=default)

    def _session_id(self, _id):
        if _id:
            df = self.df["chat_session"]
            serie = df.loc[df["user_id"] == _id]
        else:
            serie = self.df["chat_session"]
        return serie["id"].values

    def _kf_response(self, log_table):
        """
        客服回应会话
        :param log_table: chat_log
        :return:
        """
        log_table = log_table.loc[
            (log_table.source == "9")
            & (log_table.raw.str.contains('"from": "kf"'))
            & ~(log_table.raw.str.contains('"msg_type": "event"'))
            & (log_table.raw.str.contains('"is_sys_send": 0'))
            ]
        return log_table

    def _msg_table(self, log_table):
        """
        chat_log消息量
        :param log_table: chat_log
        :return:
        """
        msg_table = log_table.loc[
            (log_table.source == "9")
            & (
                    (log_table.raw.str.contains('"from": "kf"'))
                    | (log_table.raw.str.contains('"from": "user"'))
            )
            & (log_table.raw.str.contains('"is_sys_send": 0'))
            ]
        return msg_table

    def _会话量(self, table):
        """
        总的会话数
        :param table: chat_session
        :return:
        """
        num = table.loc[table.status == 2].groupby("ori_session").groups
        return len(num)

    def _接待量(self, table):
        """
        总的接待数量
        :param table: chat_session
        :return:
        """
        df = table.loc[(table.user_id.notnull()) & (table.status == 2)]
        return self._count(df)

    def _机器人接待量(self, table):
        """
        机器人会话数量
        :param table: chat_session
        :return:
        """
        df = table.loc[(table.user_id.isnull()) & (table.creator == 0)]
        return self._count(df)

    def _排队会话量(self, table):
        """
        访客进入排队
        :param table:chat_queue
        :return:
        """
        return self._count(table)

    def _人工会话量(self, table):
        """
        客服参与的会话数量
        :param table: chat_session
        :return:
        """
        df = table.loc[(table.status == 2) & (table.creator.isin([1, 9, 10, 11, 12, 13, 14, 15]))]
        return self._count(df)

    def _机器人转人工(self, table):
        """
        机器人转人工
        :param table:
        :return:
        """
        df = table.loc[(table.status == 2) & (table.user_id.notnull()) & (table.creator == 1)]
        return self._count(df)

    def _人工消息量(self, session_table, log_table):
        """
        客服被动消息量
        :param session_table:
        :param log_table:
        :return:
        """
        active_ori_sessions = set(session_table.loc[
                                      (session_table.status == 2)
                                      & (session_table.user_id.notnull())
                                      & (session_table.creator.isin([2, 3]))].ori_session.values)

        no_active_sids = set(session_table.loc[
                                 (session_table.status == 2)
                                 & (session_table.user_id.notnull())
                                 & ~(session_table.ori_session.isin(active_ori_sessions))].sid.values)

        msg = log_table.loc[
            (log_table.source == "9")
            & ~(log_table.raw.str.contains('"msg_type": "event"'))
            & (
                    (log_table.raw.str.contains('"from": "user"'))
                    | (log_table.raw.str.contains('"from": "kf"'))
            )
            & (
                (log_table.raw.str.contains('"is_sys_send": 0'))

            )
            & ~(log_table.raw.str.contains('"text": "转机器人"'))
            & (log_table.sid.isin(no_active_sids))
            ]

        visitor_msg = msg.loc[(msg.raw.str.contains('"from": "user"'))]
        kf_msg = msg.loc[(msg.raw.str.contains('"from": "kf"'))]
        return {
            "总消息量": int(msg.id.count()),
            "访客消息量": int(visitor_msg.id.count()),
            "客服消息量": int(kf_msg.id.count())
        }

    def _满意统计(self, table):
        """
        满意统计
        :param table: chat_session
        :return:
        """
        df = table.loc[(table.user_id.notnull()) & (table.status == 2)]
        df = df.sort_values('created', ascending=False).groupby('ori_session',
                                                                as_index=False).first()
        satisfaction = df["satisfaction"]
        size = satisfaction.size
        satis = satisfaction.value_counts()
        评价总量 = size
        非常满意 = satis.get(5, 0)
        满意 = satis.get(4, 0)
        一般 = satis.get(3, 0)
        不满意 = satis.get(2, 0)
        非常不满意 = satis.get(1, 0)
        未评价 = 评价总量 - (非常满意 + 满意 + 一般 + 不满意 + 非常不满意)
        if not size:
            return {"未评价": 0, "非常不满意": 0, "不满意": 0, "一般": 0, "满意": 0,
                    "非常满意": 0, "评价总量": 0}
        return {"未评价": 未评价, "非常不满意": 非常不满意, "不满意": 不满意, "一般": 一般, "满意": 满意,
                "非常满意": 非常满意, "评价总量": 评价总量}

    def _一次性解决量(self, table):
        """
        一次性解决率
        :param table: chat_session
        :return:
        """
        table = table.loc[
            (table.status == 2)
            & (table.one_solved_status == 1)
            & (table.creator.isin([1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))]
        return self._count(table)

    def _解决方式(self, table):
        """
        访客解决数量
        :param table:
        :return:
        """
        human_table = table.loc[(table.status == 2) & (table.question_status == 1) & (table.user_id.notnull())]
        human_deal = int(human_table.user_id.describe()["count"])

        # 机器人解决量
        robot_df = table.loc[
            (table.status == 2)
            & (
                    (
                            (table.creator == 会话发起方式["机器人"])
                            & (table.stop_way == 会话结束方式["机器人超时结束"])
                    )
                    | (
                            (table.creator == 会话发起方式["客户入队列"])
                            & (table.stop_way.isin([会话结束方式["放弃排队"], 会话结束方式["客服都下线出队"]]))
                    )
            )]

        return {"human": human_deal, "robot": self._count(robot_df)}

    def _用户咨询分类(self, table):
        """
        :param table: chat_session
        :return: {"tag1": "id": tag1_id, "count": 0, "sub_tags": {"id": tag2_id, "count": 0}}
        """
        # 用户分类咨询
        df = table.loc[table.user_id.notnull() & (table.status == 2)]
        df = df.sort_values('created', ascending=False).groupby('ori_session',
                                                                as_index=False).first()
        df = df.fillna(value=0)
        total_num = df["tag1"].size

        classify, num1, num2 = {"未分类": 0}, 0, 0
        for tag in df.itertuples():
            tag1_content = self.tags.get(int(tag.tag1))
            tag2_content = self.tags.get(int(tag.tag2))
            classify["分类总数"] = total_num

            if tag1_content:
                if tag1_content not in classify:
                    classify[tag1_content] = {
                        "id": int(tag.tag1),
                        "count": 1,
                        "sub_tags": {}
                    }

                else:
                    classify[tag1_content]["count"] += 1

                if tag2_content not in classify.get(tag1_content).get(
                        "sub_tags") and tag2_content:
                    sdict = {
                        tag2_content: {"id": int(tag.tag2), "count": 1}}
                    classify[tag1_content]["sub_tags"].update(sdict)
                elif tag2_content in classify.get(tag1_content).get(
                        "sub_tags") and tag2_content:
                    classify[tag1_content]["sub_tags"][tag2_content][
                        "count"] += 1
            else:
                classify["未分类"] += 1
        return classify

    def _客服会话时长(self, chat_session, chat_log):
        """
        客服已接待时长(会话段)
        :param chat_session:
        :param chat_log:
        :return: 各会话的时间比
        """
        log_sids = set(self._kf_response(chat_log).sid.values)
        reception_session = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.sid.isin(log_sids))
            & ~(chat_session.creator.isin([2, 3]))]
        session = {
            "<2m": 0,
            "2m-4m": 0,
            "4m-6m": 0,
            "6m-8m": 0,
            ">8m": 0,
            "all": 0
        }
        for each_session in reception_session.itertuples():
            _time = (each_session.stop_time - each_session.created).total_seconds()
            if _time <= 120:
                session["<2m"] += 1
            elif 120 < _time <= 240:
                session["2m-4m"] += 1
            elif 240 < _time <= 360:
                session["4m-6m"] += 1
            elif 360 < _time <= 480:
                session["6m-8m"] += 1
            elif _time > 480:
                session[">8m"] += 1
            session["all"] += _time
        return session

    def _会话时长(self, chat_session, chat_log):
        """
        客服会话时长(按会话量）
        :param chat_session:
        :param chat_log:
        :return:
        """
        log_sids = set(self._kf_response(chat_log).sid.values)
        reception_session = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.sid.isin(log_sids))
            & ~(chat_session.creator.isin([2, 3]))]
        reception_session_ori_session = set(reception_session.ori_session.values)
        rest_session = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.ori_session.isin(reception_session_ori_session))]
        session = {
            "<2m": 0,
            "2m-4m": 0,
            "4m-6m": 0,
            "6m-8m": 0,
            ">8m": 0,
            "all": 0,
            "total_count": 0
        }
        for ori_session, dataframe in rest_session.groupby("ori_session"):
            if dataframe.shape[0] != 0:
                _time = (dataframe.iloc[-1].stop_time - dataframe.iloc[0].created).total_seconds()
                if _time <= 120:
                    session["<2m"] += 1
                elif 120 < _time <= 240:
                    session["2m-4m"] += 1
                elif 240 < _time <= 360:
                    session["4m-6m"] += 1
                elif 360 < _time <= 480:
                    session["6m-8m"] += 1
                elif _time > 480:
                    session[">8m"] += 1
                session["all"] += _time
                session["total_count"] += 1
        return session

    def _客服响应时长(self, chat_session, chat_log):
        """
        客服响应时长
        :param chat_session:
        :param chat_log:
        :return: 客服 会话段的响应时长
        """
        # kf_response
        log_kf_respose = self._kf_response(chat_log)
        sids_useful = set(log_kf_respose.sid.values)

        # 找到chat_session对应sid -> created
        sid_createds = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & ~(chat_session.sid.isin([2, 3]))
            & (chat_session.sid.isin(sids_useful))]

        sid_created_mapping = {each_session.sid: each_session.created for each_session in sid_createds.itertuples()}

        # 筛选满足要求chat_session 去除主动的sid
        not_satisfy_sid = set(chat_session.loc[chat_session.creator.isin([2, 3])].sid.values)
        rest_log = chat_log.loc[chat_log.sid.isin(sids_useful) & ~(chat_log.sid.isin(not_satisfy_sid))]

        # 区分每一个sid
        kf_response = {}
        first_response = {}
        for sid, dataframe in rest_log.sort_values('created', ascending=True).groupby('sid', as_index=False):
            sid_list, only_one_kf, user_status = [], [], False
            for each_log in dataframe.itertuples():
                content = each_log.raw
                _type = re.search(r'"msg_type": "event"', content)
                _from = re.findall(r'"from": "(\w+)"', content)
                _sys = re.search(r'"is_sys_send": 0', content)
                if _type and _from and not user_status and not _sys:
                    if _from[0] == "kf":
                        if not sid_list:
                            sid_list.append({"from": "user", "time": dataframe.iloc[0].created})
                            user_status = True
                        if not only_one_kf:
                            only_one_kf.append({"from": "user", "time": dataframe.iloc[0].created})

                if not _type and _from and not user_status and _sys:
                    if _from[0] == "user":
                        sid_list.append({"from": "user", "time": each_log.created})
                        if not only_one_kf:
                            only_one_kf.append({"from": "user", "time": each_log.created})
                        user_status = True
                if user_status and not _type and _from and _sys:
                    if sid_list:
                        if sid_list[-1]["from"] != _from[0]:
                            sid_list.append({"from": "kf", "time": each_log.created})
                            user_status = False
                            if len(only_one_kf) == 1:
                                only_one_kf.append({"from": "kf", "time": each_log.created})

            if sid_list:
                if sid_list[-1]["from"] != "kf":
                    sid_list.pop(-1)
            if only_one_kf:
                if only_one_kf[-1]["from"] != "kf":
                    only_one_kf.pop(-1)

            kf_response[sid] = sid_list
            first_response[sid] = only_one_kf

        response = self._cal_response_time(kf_response)
        first_response = self._cal_response_time(first_response)
        return {
            "响应时长": response,
            "响应数": response.get("total_count", 0),
            "首次响应时长": first_response,
            "首次响应数": first_response.get("total_count", 0),
            "30s应答数": response.get("<15s", 0) + response.get("15s-30s", 0)}

    @staticmethod
    def _cal_response_time(time_data):
        """
        处理响应时长
        :return:
        """
        response = defaultdict(dict)
        for key, value in time_data.items():
            if len(value) >= 2 and value[-1]["from"] != "kf":
                value.pop(-1)
                user = value[::2]
                kf = value[1::2]
                response[key] = [
                    (each_kf["time"] - each_user["time"]).total_seconds() for each_user, each_kf in zip(user, kf)
                ]
            elif len(value) >= 2 and value[-1]["from"] == "kf":
                user = value[::2]
                kf = value[1::2]
                response_time = [(each_kf["time"] - each_user["time"]).total_seconds() for each_user, each_kf in
                                 zip(user, kf)]
                response[key] = response_time
            else:
                response[key] = []
        response_time = {
            "<15s": 0,
            "15s-30s": 0,
            "30s-45s": 0,
            "45s-1m": 0,
            ">1m": 0,
            "all": 0,
            "total_count": 0
        }
        for sid, each_sid_time in response.items():
            if each_sid_time:
                for value in each_sid_time:
                    if value <= 15:
                        response_time["<15s"] += 1
                    elif 15 < value <= 30:
                        response_time["15s-30s"] += 1
                    elif 30 < value <= 45:
                        response_time["30s-45s"] += 1
                    elif 45 < value <= 60:
                        response_time["45s-1m"] += 1
                    elif value > 60:
                        response_time[">1m"] += 1
                    response_time["total_count"] += 1
                    response_time["all"] += value

        return response_time

    def _响应时长(self, chat_session, chat_log):
        """
        响应时长
        :param table:
        :return:
        """
        # 获取有客服接待的会话 根据客服必须接待
        log_table = self._kf_response(chat_log)
        log_sids = set(log_table.sid.values)

        reception_session = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.sid.isin(log_sids))
            & ~(chat_session.creator.isin([2, 3]))]

        # 得到有客服响应的会话段 再得到整个会话
        ori_session = set(reception_session.ori_session.values)
        true_session = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.ori_session.isin(ori_session))
            & ~(chat_session.creator.isin([2, 3]))]
        true_sids = set(true_session.sid.values)

        # sid -> ori_session
        sids_ori_session = {
            tuple(set(dataframe.sid.values)): ori_session for ori_session, dataframe in
            true_session.groupby("ori_session")}

        # chat_log 消息段包括访客的消息以及客服的消息
        rest_log = chat_log.loc[chat_log.sid.isin(true_sids)]

        # 获取sid dataframe 映射
        sid_dataframe = {sid: dataframe for sid, dataframe in rest_log.sort_values(
            by=["created"], inplace=False, ascending=True).groupby("sid")}

        ori_session_dataframes = {}
        for sids, ori_session in sids_ori_session.items():
            for sid in sids:
                if ori_session not in ori_session_dataframes:
                    ori_session_dataframes[ori_session] = sid_dataframe.get(sid)
                else:
                    ori_session_dataframes[ori_session] = pd.concat(
                        [ori_session_dataframes[ori_session], sid_dataframe.get(sid)], axis=0).sort_values(
                        by=["created"], inplace=False, ascending=True)

        response = {}
        first_response = {}
        for ori_session, dataframe in ori_session_dataframes.items():
            sid_list, only_one_kf = [], []
            user_status = False
            for each_log in dataframe.itertuples():
                content = each_log.raw
                _type = re.search(r'"msg_type": "event"', content)
                _from = re.findall(r'"from": "(\w+)"', content)
                _sys = re.search(r'"is_sys_send": 0', content)
                if _type and not user_status and _from and not _sys:
                    if _from[0] == "kf":
                        sid_list.append({"from": "user", "time": each_log.created})
                        if not only_one_kf:
                            only_one_kf.append({"from": "user", "time": each_log.created})
                        user_status = True
                if not _type and _from and not user_status and _sys:
                    if _from[0] == "user":
                        sid_list.append({"from": "user", "time": each_log.created})
                        if not only_one_kf:
                            only_one_kf.append({"from": "user", "time": each_log.created})
                        user_status = True
                if user_status and not _type and _from and _sys:
                    if sid_list:
                        if sid_list[-1]["from"] != _from[0]:
                            sid_list.append({"from": "kf", "time": each_log.created})
                            user_status = False
                            if len(only_one_kf) == 1:
                                only_one_kf.append({"from": "kf", "time": each_log.created})

            if sid_list:
                if sid_list[-1]["from"] != "kf":
                    sid_list.pop(-1)
            if only_one_kf:
                if only_one_kf[-1]["from"] != "kf":
                    only_one_kf.pop(-1)

            response[ori_session] = sid_list
            first_response[ori_session] = only_one_kf

        response = self._cal_response_time(response)
        first_response = self._cal_response_time(first_response)
        return {
            "响应时长": response,
            "响应数": response.get("total_count", 0),
            "首次响应时长": first_response,
            "首次响应数": first_response.get("total_count", 0),
            "30s应答数": response.get("<15s", 0) + response.get("15s-30s", 0)}

    def _机器人转人工会话量(self, table):
        """
        机器人转人工会话量
        :param table: chat_session
        :return:
        """
        df = table.loc[table.user_id.notnull() & (table["creator"] == 1) & (table["status"] == 2)]
        return self._count(df)

    def _访客统计_用户来源(self, table):
        """
        访客来源
        :param table: chat_session 与 chat_user
        :return:
        """
        session_df = table.get("session")
        总访客, 用户来源 = [], {"h5": [], "weixin": []}
        session_df = session_df.loc[
            (session_df.last_session.isnull()) & (session_df.user_id.isnull()) & (session_df["status"] == 2)]
        for uid, dataframe in session_df.groupby("uid"):
            总访客.append(uid)
            sess = self._series_index(dataframe, 0)
            用户来源[sess.get("source")].append(uid)

        新访客 = []
        for row in table.get("user").itertuples():
            新访客.append(row.uid)
        return self._json_dumps({"总访客": 总访客, "新访客": 新访客}), self._json_dumps(用户来源)

    def _进入排队人数(self, table):
        """
        获取排过队的人数
        :param table: chat_queue
        :return: nums
        """
        if table.shape[0] == 0:
            return 0
        else:
            table = table.loc[table.status == 2]
            return self._count(table)

    def _转人工数量(self, table):
        """
        获取转人工解决量
        :param table: chat_session
        :return: 转人工的次数
        """
        if table.shape[0] == 0:
            return 0
        robot_df = table.loc[(table.status == 2) & (table.user_id.isnull())]
        nums = 0
        for each_robot_msg in robot_df.itertuples():
            is_to_kf_status = json.loads(each_robot_msg.ext_bi) if each_robot_msg.ext_bi else {}
            if is_to_kf_status and "to_kf" in is_to_kf_status:
                if is_to_kf_status.get("to_kf") == 1:
                    nums += 1
        return nums

    def _排队进人工会话量(self, table):
        """
        通过排队最终进入人工会话的量
        :param table: table: chat_session
        :return: nums
        """
        if table.shape[0] == 0:
            return 0
        else:
            table = table.loc[
                (table.status == 2)
                & (table.creator == 8)
                & (table.stop_way.isin([14, 15, 17, 19, 20]))]
            return self._count(table)

    def _服务总时长(self, table, now, statis_type):
        """
        客服服务时长
        :param table: chat_session
        :return: times (s)
        """
        start_time = now - datetime.timedelta(days=1) if statis_type == 1 else now - datetime.timedelta(hours=1)
        end_time = now
        table = table.loc[
            (
                    (table.user_id.notnull())
                    & (table.status == 2)
            ) & (table.creator.isin(
                [
                    会话发起方式["客户转人工"],
                    会话发起方式["客服转交客服"],
                    会话发起方式["客服转交客服组"],
                    会话发起方式["客服超时转交"],
                    会话发起方式["客服强制下线转交"],
                    会话发起方式["队列自动接入人工"],
                    会话发起方式["队列手动接入人工"],
                    会话发起方式["队列指派接入人工"],
                    会话发起方式["关闭队列自动接入人工"],
                    会话发起方式["队列指派到客服组"],
                    会话发起方式["机器人转交客服"],
                    会话发起方式["机器人转交客服组"],
                    会话发起方式["强制转交给客服"],
                    会话发起方式["强制转交给客服组"],
                    会话发起方式["客服被动下线转交"]
                ]))
            ]
        user_data = defaultdict(list)
        for user_id, data in table.groupby("user_id"):
            for each in data.itertuples():
                created = pd.to_datetime(each.created)
                stop_time = pd.to_datetime(each.stop_time)
                user_data[int(user_id)].append([created, stop_time])
        haved_session_user = {}
        for user, user_content in user_data.items():
            haved_session_user[user] = self.cloc_service_time(user_content)
        return haved_session_user

    def _总独立接待量(self, chat_session, chat_log):
        """
        客服数据总览-独立接待量
        :param table: chat_session_ext
        :return:
        """
        ori_session_ids = []
        chat_session = chat_session.loc[chat_session.user_id.notnull()]
        for ori_session, dataframe in chat_session.groupby("ori_session"):
            if dataframe.iloc[0].creator in [1, 9, 10, 11, 12, 13, 14, 15] and dataframe.shape[0] == 1:
                if dataframe.iloc[0].stop_way not in [9, 10, 23, 24, 25, 26, 28]:
                    ori_session_ids.append(ori_session)
        rest_session = chat_session.loc[chat_session.ori_session.isin(ori_session_ids)]

        log_table = chat_log.loc[
            (chat_log.source == "9")
            & (chat_log.raw.str.contains('"from": "kf"'))
            & (chat_log.raw.str.contains('"is_sys_send": 0'))
            & (~chat_log.raw.str.contains('"msg_type": "event"'))
            ]
        log_sids = set(log_table.sid.values)
        session_table = rest_session.loc[rest_session.sid.isin(log_sids)]

        count = 0
        for ori_session, dataframe in session_table.groupby("ori_session"):
            count += 1
        return count

    def _总接待会话量(self, chat_session, chat_log):
        """
        客服数据总览-接待会话量
        :param chat_session: chat_session
        :param chat_log: chat_log
        :return:
        """
        ori_session_ids = []
        for ori_session, dataframe in chat_session.groupby("ori_session"):
            if dataframe.iloc[0].creator in [1, 9, 10, 11, 12, 13, 14, 15]:
                ori_session_ids.append(ori_session)
        rest_session = chat_session.loc[chat_session.ori_session.isin(ori_session_ids)]

        log_table = chat_log.loc[
            (chat_log.source == "9")
            & (chat_log.raw.str.contains('"from": "kf"'))
            & (chat_log.raw.str.contains('"is_sys_send": 0'))
            & (~chat_log.raw.str.contains('"msg_type": "event"'))
            ]
        log_sids = set(log_table.sid.values)
        session_table = rest_session.loc[rest_session.sid.isin(log_sids)]
        count = 0
        for ori_session, dataframe in session_table.groupby("ori_session"):
            count += 1
        return count

    def _接入会话量(self, table):
        """
        客服工作量-接入会话量
        :param table: chat_session
        :return:
        """
        df = table.loc[
            (table.status == 2)
            & (table.user_id.notnull())
            & (table.creator.isin(
                [
                    会话发起方式["客户转人工"],  # 1
                    会话发起方式["客服转交客服"],  # 4
                    会话发起方式["客服转交客服组"],  # 5
                    会话发起方式["客服超时转交"],  # 6
                    会话发起方式["客服强制下线转交"],  # 7
                    会话发起方式["队列自动接入人工"],  # 9
                    会话发起方式["队列手动接入人工"],  # 10
                    会话发起方式["队列指派接入人工"],  # 11
                    会话发起方式["关闭队列自动接入人工"],  # 12
                    会话发起方式["队列指派到客服组"],  # 13
                    会话发起方式["机器人转交客服"],  # 14
                    会话发起方式["机器人转交客服组"],  # 15
                    会话发起方式["强制转交给客服"],  # 16
                    会话发起方式["强制转交给客服组"],  # 17
                    会话发起方式["客服被动下线转交"]  # 18
                ])
            )]
        return self._count(df)

    def _接待会话量(self, chat_session, chat_log):
        """
        客服工作量-接待会话量
        :param table: chat_session, chat_log
        :return:
        """
        ori_session_ids = []
        for ori_session, dataframe in chat_session.groupby("ori_session"):
            if dataframe.iloc[0].creator in [1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                ori_session_ids.append(ori_session)
        rest_session = chat_session.loc[chat_session.ori_session.isin(ori_session_ids)]

        log_table = chat_log.loc[
            (chat_log.source == "9")
            & (chat_log.raw.str.contains('"from": "kf"'))
            & (chat_log.raw.str.contains('"is_sys_send": 0'))
            & (~chat_log.raw.str.contains('"msg_type": "event"'))
            ]
        log_sids = set(log_table.sid.values)
        session_table = rest_session.loc[rest_session.sid.isin(log_sids)]
        count = 0
        for ori_session, dataframe in session_table.groupby("ori_session"):
            count += 1
        return count

    def _独立接待量(self, chat_session, chat_log):
        """
        客服工作量-独立接待量
        :param table: chat_session chat_log
        :return:
        """
        ori_session_ids = []
        chat_session = chat_session.loc[chat_session.creator != 8]
        for ori_session, dataframe in chat_session.groupby("ori_session"):
            if dataframe.iloc[0].creator in [1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] and \
                    dataframe.shape[0] == 1:
                if dataframe.iloc[0].stop_way not in [9, 10, 23, 24, 25, 26, 28]:
                    ori_session_ids.append(ori_session)
        rest_session = chat_session.loc[chat_session.ori_session.isin(ori_session_ids)]

        log_table = chat_log.loc[
            (chat_log.source == "9")
            & (chat_log.raw.str.contains('"from": "kf"'))
            & (chat_log.raw.str.contains('"is_sys_send": 0'))
            & (~chat_log.raw.str.contains('"msg_type": "event"'))
            ]
        log_sids = set(log_table.sid.values)
        session_table = rest_session.loc[rest_session.sid.isin(log_sids)]

        count = 0
        for ori_session, dataframe in session_table.groupby("ori_session"):
            count += 1
        return count

    def _首次未响应会话量(self, table):
        """
        客服工作量-首次未响应会话量
        :param table: chat_session
        :return:
        """
        df = table.loc[
            (table.status == 2)
            & (table.user_id.notnull())
            & (table.stop_way.isin(
                [
                    会话结束方式["客服超时结束"],
                    会话结束方式["客服超时转交"]
                ]
            ))
            ]
        return self._count(df)

    def _结束会话量(self, chat_session, chat_log):
        """
        客服工作量-结束会话量
        :param table: chat_session
        :return:
        """
        log_table = chat_log.loc[
            (chat_log.source == "9")
            & (chat_log.raw.str.contains('"from": "kf"'))
            & (chat_log.raw.str.contains('"is_sys_send": 0'))
            & (~chat_log.raw.str.contains('"msg_type": "event"'))
            ]
        log_sids = set(log_table.sid.values)
        df = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.sid.isin(log_sids))
            & (chat_session.creator.isin([
                会话发起方式["客户转人工"],  # 1
                会话发起方式["客服转交客服"],  # 4
                会话发起方式["客服转交客服组"],  # 5
                会话发起方式["客服超时转交"],  # 6
                会话发起方式["客服强制下线转交"],  # 7
                会话发起方式["队列自动接入人工"],  # 9
                会话发起方式["队列手动接入人工"],  # 10
                会话发起方式["队列指派接入人工"],  # 11
                会话发起方式["关闭队列自动接入人工"],  # 12
                会话发起方式["队列指派到客服组"],  # 13
                会话发起方式["机器人转交客服"],  # 14
                会话发起方式["机器人转交客服组"],  # 15
                会话发起方式["强制转交给客服"],  # 16
                会话发起方式["强制转交给客服组"],  # 17,
                会话发起方式["客服被动下线转交"],  # 18
            ]))
            & (chat_session.stop_way.isin(
                [
                    会话结束方式["客服超时结束"],
                    会话结束方式["客服下线"],
                    会话结束方式["用户手动"],
                    会话结束方式["客服手动"],
                    会话结束方式["用户超时结束"],
                    会话结束方式["客服被动下线"],
                    会话结束方式["客服强制下线结束"],

                ]))
            ]
        return self._count(df)

    def _主动会话量(self, table):
        """
        客服工作量-主动会话量
        :param table:
        :return:
        """
        df = table.loc[
            (table.status == 2)
            & (table.user_id.notnull())
            & (table.creator.isin(
                [
                    会话发起方式["客服激活"],
                    会话发起方式["客服工单发起"]
                ]
            ))]
        return self._count(df)

    def _主动转接量(self, table):
        """
        客服工作量-转接量
        :param table: chat_session
        :return:
        """
        df = table.loc[
            (table.status == 2)
            & (table.user_id.notnull())
            & (table.creator.isin([

                会话发起方式["客户转人工"],  # 1
                会话发起方式["客服转交客服"],  # 4
                会话发起方式["客服转交客服组"],  # 5
                会话发起方式["客服超时转交"],  # 6
                会话发起方式["客服强制下线转交"],  # 7
                会话发起方式["队列自动接入人工"],  # 9
                会话发起方式["队列手动接入人工"],  # 10
                会话发起方式["队列指派接入人工"],  # 11
                会话发起方式["关闭队列自动接入人工"],  # 12
                会话发起方式["队列指派到客服组"],  # 13
                会话发起方式["机器人转交客服"],  # 14
                会话发起方式["机器人转交客服组"],  # 15
                会话发起方式["强制转交给客服"],  # 16
                会话发起方式["强制转交给客服组"],  # 17,
                会话发起方式["客服被动下线转交"],  # 18
            ]))
            & (table.stop_way.isin([
                会话结束方式["客服手动转交"],
                会话结束方式["客服转交客服组"]
            ]))
            ]
        return self._count(df)

    def _接待总时长(self, table):
        """
        客服接待总时长
        :param table: chat_session
        :return:
        """
        table = table.loc[
            (table.status == 2)
            & (table.user_id.notnull())
            & (table.creator.isin([1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))
            ]
        reception_time = 0
        for each_session in table.itertuples():
            reception_time += (each_session.stop_time - each_session.created).total_seconds()
        return reception_time

    def _未回复会话量_无效会话量(self, chat_session, chat_log):
        """
        未回复会话量
        :param chat_session:
        :param chat_log:
        :return:
        """
        to_kf = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.creator.isin(
                [
                    会话发起方式["客户转人工"],  # 1
                    会话发起方式["客服转交客服"],  # 4
                    会话发起方式["客服转交客服组"],  # 5
                    会话发起方式["客服超时转交"],  # 6
                    会话发起方式["客服强制下线转交"],  # 7
                    会话发起方式["队列自动接入人工"],  # 9
                    会话发起方式["队列手动接入人工"],  # 10
                    会话发起方式["队列指派接入人工"],  # 11
                    会话发起方式["关闭队列自动接入人工"],  # 12
                    会话发起方式["队列指派到客服组"],  # 13
                    会话发起方式["机器人转交客服"],  # 14
                    会话发起方式["机器人转交客服组"],  # 15
                    会话发起方式["强制转交给客服"],  # 16
                    会话发起方式["强制转交给客服组"],  # 17
                    会话发起方式["客服被动下线转交"]  # 18
                ])
            )]
        total_sids = set(to_kf.sid.values)

        visit = chat_log.loc[
            (chat_log.source == "9")
            & (chat_log.raw.str.contains('"from": "user"'))
            & (chat_log.raw.str.contains('"is_sys_send": 0'))
            & ~(chat_log.raw.str.contains('"text": "转机器人"'))
            & ~(chat_log.raw.str.contains('"msg_type": "event"'))]
        visit_sids = set(visit.sid.values) & total_sids

        kf = chat_log.loc[
            (chat_log.source == "9")
            & (chat_log.raw.str.contains('"from": "kf"'))
            & (chat_log.raw.str.contains('"is_sys_send": 0'))
            & ~(chat_log.raw.str.contains('"msg_type": "event"'))]
        kf_sids = set(kf.sid.values) & total_sids
        return len(total_sids - kf_sids), len(total_sids - visit_sids)

    def _问题解决_问题未解决(self, chat_session):
        """
        客服问题解决
        :param chat_session:
        :return:
        """
        solve = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.creator.isin(
                [
                    会话发起方式["客户转人工"],  # 1
                    会话发起方式["客服转交客服"],  # 4
                    会话发起方式["客服转交客服组"],  # 5
                    会话发起方式["客服超时转交"],  # 6
                    会话发起方式["客服强制下线转交"],  # 7
                    会话发起方式["队列自动接入人工"],  # 9
                    会话发起方式["队列手动接入人工"],  # 10
                    会话发起方式["队列指派接入人工"],  # 11
                    会话发起方式["关闭队列自动接入人工"],  # 12
                    会话发起方式["队列指派到客服组"],  # 13
                    会话发起方式["机器人转交客服"],  # 14
                    会话发起方式["机器人转交客服组"],  # 15
                    会话发起方式["强制转交给客服"],  # 16
                    会话发起方式["强制转交给客服组"],  # 17
                    会话发起方式["客服被动下线转交"]  # 18
                ]
            ))
            & (chat_session.question_status == 1)]
        no_solve = chat_session.loc[
            (chat_session.status == 2)
            & (chat_session.user_id.notnull())
            & (chat_session.creator.isin(
                [
                    会话发起方式["客户转人工"],  # 1
                    会话发起方式["客服转交客服"],  # 4
                    会话发起方式["客服转交客服组"],  # 5
                    会话发起方式["客服超时转交"],  # 6
                    会话发起方式["客服强制下线转交"],  # 7
                    会话发起方式["队列自动接入人工"],  # 9
                    会话发起方式["队列手动接入人工"],  # 10
                    会话发起方式["队列指派接入人工"],  # 11
                    会话发起方式["关闭队列自动接入人工"],  # 12
                    会话发起方式["队列指派到客服组"],  # 13
                    会话发起方式["机器人转交客服"],  # 14
                    会话发起方式["机器人转交客服组"],  # 15
                    会话发起方式["强制转交给客服"],  # 16
                    会话发起方式["强制转交给客服组"],  # 17
                    会话发起方式["客服被动下线转交"]  # 18
                ]
            ))
            & (chat_session.question_status == 0)]
        return self._count(solve), self._count(no_solve)


    @staticmethod
    def cloc_service_time(time_tuple_list):
        '''时间二元组列表, [(开始时间, 结束时间)]'''
        total_seconds = 0
        last_time = ""
        for start, end in time_tuple_list:
            if not last_time:
                total_seconds += (end - start).total_seconds()
            else:
                if last_time > start:
                    total_seconds += (end - last_time).total_seconds()
                elif last_time <= start:
                    total_seconds += (end - start).total_seconds()
            last_time = end
        return total_seconds if total_seconds >= 0 else 0


if __name__ == "__main__":
    from utils.utils import get_db
    db = get_db()
    df = {}
    start = "2018-12-05"
    end = "2018-12-25"

    df["chat_session"] = pd.read_sql_query(
        """
        select c_s2.* from chat_session as c_s2 where c_s2.ori_session in 
        (select c_s1.ori_session  from chat_session as c_s1 
        where c_s1.created > "{}" 
        and c_s1.created < "{}" and c_s1.sid = c_s1.ori_session)
        """.format(start, end), db.bind)
    df["chat_log"] = pd.read_sql_query(
        """
        select c_l.* from chat_log as c_l join 
        (select * from chat_session as c_s2 where c_s2.ori_session in 
        (select c_s1.ori_session  from chat_session as c_s1 
        where c_s1.created > "{}" and c_s1.created < "{}" and c_s1.sid = c_s1.ori_session)) as c_s
        on c_l.sid = c_s.sid and c_l.uid = c_s.uid where c_l.created > "{}"
        """.format(start, end, start), db.bind
    )
    df["chat_user"] = pd.read_sql_query(
        """
        select * from chat_user where created > '{}' and created <= '{}'
        """.format(start, end), db.bind)
    df['question_tag'] = pd.read_sql_table('question_tag', db.bind)
    df['users'] = pd.read_sql_table("user", db.bind)
    df["kf_status"] = pd.read_sql_query(
        "select * from company_setting where name = 'kf_status'", db.bind)
    df["chat_queue"] = pd.read_sql_query(
        "select * from chat_queue where start_time > '{}' and start_time <= '{}'".format(
            start, end), db.bind)

    for key, value in df.items():
        df[key] = value.loc[value.cid.isin([149, "149"])]

    # result = BaseStatis(df, statis_type=1)._会话时长(df["chat_session"], df["chat_log"])
    # result1 = BaseStatis(df, statis_type=1)._客服响应时长(df["chat_session"], df["chat_log"])
    # pprint(result1)
    # print()
    # result1 = BaseStatis(df, statis_type=1)._结束会话量(df["chat_session"], df["chat_log"])
    # result2 = BaseStatis(df, statis_type=1)._响应时长(df["chat_session"], df["chat_log"])
    # pprint(result2)

    base = BaseStatis(df, statis_type=1)

    db.close()
