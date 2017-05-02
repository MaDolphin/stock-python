import pymysql
import numpy as np


def inset_db(stock_id, date, true_data, prediction_data):
    conn = pymysql.connect(
        host='121.42.197.31',
        port=3306,
        user='root',
        passwd='291910',
        db='stock',
    )
    cur = conn.cursor()

    sql_data = prepare_date0(stock_id, date, true_data, prediction_data)
    print(sql_data)
    # 一次插入多条记录
    # sqli = "insert into prediction values(%s, %s, %s, %s, %s)"
    sqli = "insert into prediction values(?, ?, ?, ?, ?)"
    cur.executemany(sqli, sql_data)

    cur.close()
    conn.commit()
    conn.close()


def prepare_date(stock_id, date, true_data, prediction_data):
    sql_data = []
    for i in range(len(true_data)):
        data = (stock_id, date, 0, true_data[i], i)
        sql_data.append(data)
    for i in range(len(prediction_data)):
        data = (stock_id, date, 1, prediction_data[i], i)
        sql_data.append(data)
    return sql_data


def prepare_date0(stock_id, date, true_data, prediction_data):
    data_true = ""
    data_prediction = ""
    for i in range(len(true_data)):
        if i == 0:
            data_true = "(" + stock_id + "," + date + "," + str(0) + "," + str(true_data[i]) + "," + str(i) + ")"
        else:
            data_true = data_true + "," + "(" + stock_id + "," + date + "," + str(0) + "," + str(
                true_data[i]) + "," + str(i) + ")"

    for i in range(len(prediction_data)):
        if i == 0:
            data_prediction = "(" + stock_id + "," + date + "," + str(1) + "," + str(prediction_data[i]) + "," + str(
                i) + ")"
        else:
            data_prediction = data_prediction + "," + "(" + stock_id + "," + date + "," + str(1) + "," + str(
                prediction_data[i]) + "," + str(i) + ")"

    return "[" + data_true + "," + data_prediction + "]"
