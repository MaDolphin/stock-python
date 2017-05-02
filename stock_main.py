import stock_training as st
import stock_prediction as sp
import time
from SQL_Dao import inset_db


def execute_ml(stock_id):
    flag_ml, save_file_path = st.train_lstm(stock_id=stock_id)
    if flag_ml == "success":
        true_data, prediction_data = sp.prediction(stock_id=stock_id, save_file_path=save_file_path)
        date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        flag_db = inset_db(stock_id=stock_id, date=date, true_data=true_data, prediction_data=prediction_data)
        if flag_db == "success":
            return "success"
        else:
            return "fail"
    else:
        return "fail"

# execute_ml(stock_id='601766')