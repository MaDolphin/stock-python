import stock_training as st
import stock_prediction as sp
import time
from SQL_Dao import inset_db
import uuid


def execute_ml(stock_id):

    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    rnn_name = str(uuid.uuid1())

    flag_ml, save_file_path = st.train_lstm(stock_id=stock_id, rnn_name=rnn_name)
    if flag_ml == "success":
        date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        true_data, prediction_data = sp.prediction(stock_id=stock_id, save_file_path=save_file_path, rnn_name=rnn_name)
        flag_db = inset_db(stock_id=stock_id, date=date, true_data=true_data, prediction_data=prediction_data)
        if flag_db == "success":
            return "success"
        else:
            return "fail"
    else:
        return "fail"

# execute_ml(stock_id='600000')