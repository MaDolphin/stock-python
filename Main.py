import threading
import time
import Global_Queue as gq
import Receiver as receiver
import Sender as sender


class Worker(threading.Thread):
    def run(self):
        while True:
            if not gq.SHARE_Q.empty():
                item = gq.SHARE_Q.get()  # 获得任务
                print("[Processing]:", item)
                import stock_main as sm
                flag_ml = sm.execute_ml(stock_id=item)
                if flag_ml == "success":
                    sender.send("[Success]:" + item)
                else:
                    sender.send("[Fail]:" + item)


class Receiver(threading.Thread):
    def run(self):
        receiver.receive()


def main():
    r = Receiver()
    w = Worker()
    w.start()
    r.start()


if __name__ == '__main__':
    main()
