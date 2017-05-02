import pymysql


def inset_db(stock_id, date, true_data, prediction_data):
    conn = pymysql.connect(
        host='121.42.197.31',
        port=3306,
        user='root',
        passwd='291910',
        db='stock',
    )
    cur = conn.cursor()

    sql_data = prediction_data(stock_id, date, true_data, prediction_data)
    # 一次插入多条记录
    sqli = "insert into prediction(stockId, date, type, value, order) values(%s, %s, %s, %s, %s)"
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

